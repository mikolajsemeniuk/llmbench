package metrics

import (
	"context"
	"fmt"
)

// NLI computes the entailment probability of a (premise, hypothesis)
// sentence pair via the Python /nli endpoint of `cmd/modelsrv`
// (cross-encoder/nli-deberta-v3-small by default, see NLI_MODEL).
//
// This is the go/no-go probe from improvements.txt section 6: LGS's
// cosine grounding is a semantic-proximity signal, not entailment, and
// consistency (LGS's worst dimension) is precisely the dimension an
// entailment signal should help with, since paraphrase keeps cosine
// high but a genuine contradiction or unsupported claim should not
// keep the entailment probability high. cmd/nliscorer uses this
// client to build a metric structurally identical to LGS (mean over
// candidate sentences of the max score against any source sentence)
// but with entailment probability in place of embedding cosine, so
// the two are a controlled comparison of signal type holding
// aggregation fixed.
type NLI struct {
	Server *ModelServer
}

func NewNLI(host string) *NLI {
	return &NLI{Server: NewModelServer(host)}
}

// Score returns P(entailment) for premise entailing hypothesis --
// i.e. how well the premise (typically a source sentence) supports
// the hypothesis (typically a candidate sentence).
func (n *NLI) Score(ctx context.Context, premise, hypothesis string) (float64, error) {
	in := modelServerRequest{
		Reference: premise,
		Candidate: hypothesis,
	}
	res, err := n.Server.post(ctx, "/nli", in)
	if err != nil {
		return 0, fmt.Errorf("nli: %w", err)
	}
	return res.Score, nil
}
