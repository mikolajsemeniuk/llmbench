package llmbench

import (
	"context"
	"fmt"
	"io"
	"runtime"
	"sort"
	"sync"
	"sync/atomic"
	"time"
)

// Scorer evaluates one sample (candidate vs. its references) and returns a
// single aggregated score. Implementations encapsulate the per-reference fan
// out so that every metric — pure, network, or model-backed — is interchangeable.
type Scorer func(ctx context.Context, sample Sample) (float64, error)

// Result is the outcome of running one scorer across the whole dataset.
// Errors counts samples whose scoring call returned a non-nil error — those
// samples are excluded from the correlation computation so that a handful of
// network hiccups (or Ctrl+C mid-run) doesn't pollute the reported rank.
type Result struct {
	Name        string
	Scores      []float64
	Correlation Correlation
	Duration    time.Duration
	Errors      int
}

// Runner evaluates multiple scorers concurrently against a shared dataset.
type Runner struct {
	Dataset  []Sample
	Scorers  map[string]Scorer
	Workers  int       // concurrent samples per scorer; 0 falls back to runtime.NumCPU()
	Progress io.Writer // nil disables the live tracker
}

// Run executes every registered scorer and returns results sorted by name.
func (r *Runner) Run(ctx context.Context) []Result {
	workers := r.Workers
	if workers <= 0 {
		workers = runtime.NumCPU()
	}

	track := newTracker(r.Scorers, len(r.Dataset))
	stop := make(chan struct{})
	trackerDone := make(chan struct{})
	if r.Progress != nil {
		go func() {
			defer close(trackerDone)
			track.loop(r.Progress, stop)
		}()
	}

	var (
		wg      sync.WaitGroup
		mu      sync.Mutex
		results = make([]Result, 0, len(r.Scorers))
	)
	for name, scorer := range r.Scorers {
		wg.Go(func() {
			res := runScorer(ctx, name, scorer, r.Dataset, workers, track)
			mu.Lock()
			results = append(results, res)
			mu.Unlock()
		})
	}
	wg.Wait()

	if r.Progress != nil {
		close(stop)
		<-trackerDone // wait for the render goroutine to exit before the final draw
		track.draw(r.Progress)
	}

	sort.Slice(results, func(i, j int) bool { return results[i].Name < results[j].Name })
	return results
}

func runScorer(ctx context.Context, name string, scorer Scorer, dataset []Sample, workers int, track *tracker) Result {
	scores := make([]float64, len(dataset))
	ok := make([]bool, len(dataset))
	start := time.Now()
	sem := make(chan struct{}, workers)

	var (
		wg       sync.WaitGroup
		okCount  atomic.Int64
		errCount atomic.Int64
	)

dispatch:
	for i := range dataset {
		// Honour cancellation while waiting for a worker slot — stops us from
		// spawning thousands of goroutines that would only immediately fail
		// after Ctrl+C.
		select {
		case sem <- struct{}{}:
		case <-ctx.Done():
			break dispatch
		}
		wg.Go(func() {
			defer func() { <-sem }()
			s, err := scorer(ctx, dataset[i])
			var nOk, nErr int64
			if err != nil {
				nErr = errCount.Add(1)
				nOk = okCount.Load()
			} else {
				scores[i] = s
				ok[i] = true
				nOk = okCount.Add(1)
				nErr = errCount.Load()
			}
			track.update(name, int(nOk), int(nErr), time.Since(start))
		})
	}
	wg.Wait()

	// Compute correlation only on samples that actually succeeded; failed
	// samples have scores[i] == 0 which would otherwise bias the rank toward
	// the zero floor and make a fully-failed metric indistinguishable from
	// a legitimately uninformative one.
	okSamples := make([]Sample, 0, len(dataset))
	okScores := make([]float64, 0, len(dataset))
	for i, succeeded := range ok {
		if succeeded {
			okSamples = append(okSamples, dataset[i])
			okScores = append(okScores, scores[i])
		}
	}

	return Result{
		Name:        name,
		Scores:      scores,
		Correlation: NewCorrelation(okSamples, okScores),
		Duration:    time.Since(start),
		Errors:      int(errCount.Load()),
	}
}

// Sync adapts a pure math metric into a Scorer by aggregating across references.
func Sync(fn func(reference, candidate string) float64, norm Norm) Scorer {
	return func(_ context.Context, s Sample) (float64, error) {
		scores := make([]float64, len(s.References))
		for i, ref := range s.References {
			scores[i] = fn(ref, s.Candidate)
		}
		return norm(scores), nil
	}
}

// Async adapts a model-backed scoring method into a Scorer and fans references
// out in parallel goroutines — one per reference — to hide network latency.
func Async(fn func(ctx context.Context, reference, candidate string) (float64, error), norm Norm) Scorer {
	return func(ctx context.Context, s Sample) (float64, error) {
		return AggregateAsync(ctx, s.References, s.Candidate, norm, fn)
	}
}

// AggregateAsync invokes fn concurrently for every reference, then reduces the
// per-reference scores with norm. Exported so callers can wire up scorers with
// non-standard signatures (e.g. G-Eval's extra question argument).
func AggregateAsync(ctx context.Context, refs []string, cand string, norm Norm,
	fn func(ctx context.Context, reference, candidate string) (float64, error)) (float64, error) {
	if len(refs) == 0 {
		return 0, nil
	}

	scores := make([]float64, len(refs))
	var (
		wg    sync.WaitGroup
		mu    sync.Mutex
		first error
	)
	for i, ref := range refs {
		wg.Go(func() {
			v, err := fn(ctx, ref, cand)
			if err != nil {
				mu.Lock()
				if first == nil {
					first = err
				}
				mu.Unlock()
				return
			}
			scores[i] = v
		})
	}
	wg.Wait()
	if first != nil {
		return 0, first
	}
	return norm(scores), nil
}

// ---- live progress tracker ----

type tracker struct {
	mu        sync.Mutex
	names     []string
	entries   map[string]*entry
	total     int
	lastLines int
}

type entry struct {
	ok       int
	errors   int
	duration time.Duration
}

func newTracker(scorers map[string]Scorer, total int) *tracker {
	names := make([]string, 0, len(scorers))
	entries := make(map[string]*entry, len(scorers))
	for name := range scorers {
		names = append(names, name)
		entries[name] = &entry{}
	}
	sort.Strings(names)
	return &tracker{names: names, entries: entries, total: total}
}

func (t *tracker) update(name string, ok, errors int, d time.Duration) {
	t.mu.Lock()
	e := t.entries[name]
	e.ok = ok
	e.errors = errors
	e.duration = d
	t.mu.Unlock()
}

func (t *tracker) loop(w io.Writer, stop <-chan struct{}) {
	tick := time.NewTicker(250 * time.Millisecond)
	defer tick.Stop()
	for {
		select {
		case <-tick.C:
			t.draw(w)
		case <-stop:
			return
		}
	}
}

func (t *tracker) draw(w io.Writer) {
	t.mu.Lock()
	defer t.mu.Unlock()
	if t.lastLines > 0 {
		fmt.Fprintf(w, "\x1b[%dA\x1b[J", t.lastLines)
	}
	for _, name := range t.names {
		e := t.entries[name]
		errs := ""
		if e.errors > 0 {
			errs = fmt.Sprintf(" (%de)", e.errors)
		}
		fmt.Fprintf(w, "%-14s %6d/%-6d%-8s %12s\n",
			name, e.ok, t.total, errs,
			e.duration.Truncate(time.Millisecond))
	}
	t.lastLines = len(t.names)
}
