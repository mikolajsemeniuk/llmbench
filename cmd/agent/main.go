package main

import (
	"bufio"
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/exec"
	"os/signal"
	"strings"
	"time"

	adapter "github.com/i2y/langchaingo-mcp-adapter"
	mcpclient "github.com/mark3labs/mcp-go/client"
	"github.com/tmc/langchaingo/agents"
	"github.com/tmc/langchaingo/callbacks"
	"github.com/tmc/langchaingo/llms/ollama"
	"github.com/tmc/langchaingo/tools"
)

type DateTime struct{}

func (DateTime) Name() string { return "datetime" }
func (DateTime) Description() string {
	return "Returns current date and time. Input: any string, e.g. 'now'."
}
func (DateTime) Call(_ context.Context, _ string) (string, error) {
	return time.Now().Format("2006-01-02 15:04:05 MST"), nil
}

type Shell struct{}

func (Shell) Name() string { return "shell" }
func (Shell) Description() string {
	return "Runs a shell command on the host machine. Input: the command to run, e.g. 'ls -la' or 'cat file.txt'. Returns stdout and stderr."
}
func (Shell) Call(_ context.Context, input string) (string, error) {
	input = strings.TrimSpace(input)
	input = strings.Trim(input, "`\"'")
	input = strings.TrimSpace(input)

	out, err := exec.Command("bash", "-c", input).CombinedOutput()
	if err != nil {
		return string(out), fmt.Errorf("shell: %w\n%s", err, out)
	}
	s := string(out)
	if len(s) > 4000 {
		s = s[:4000] + "\n... (truncated)"
	}
	return s, nil
}

func main() {
	var (
		host    string
		model   string
		mcpURL  string
		maxIter int
		query   string
		verbose bool
	)

	flag.StringVar(&host, "host", "http://localhost:11434", "Ollama host URL")
	flag.StringVar(&model, "model", "qwen2.5-coder:7b", "Ollama model name")
	flag.StringVar(&mcpURL, "mcp", "http://localhost:9090/mcp", "MCP server URL")
	flag.IntVar(&maxIter, "max-iter", 5, "max agent reasoning iterations")
	flag.StringVar(&query, "query", "", "single query (skip for interactive REPL)")
	flag.BoolVar(&verbose, "verbose", false, "print agent reasoning steps")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	mcp, err := mcpclient.NewStreamableHttpClient(mcpURL)
	if err != nil {
		log.Fatalf("mcp client: %v", err)
	}
	defer mcp.Close()

	log.Printf("connected to MCP server at %s", mcpURL)
	a, err := adapter.New(mcp)
	if err != nil {
		log.Fatalf("mcp adapter: %v", err)
	}

	mcpTools, err := a.Tools()
	if err != nil {
		log.Fatalf("mcp tools: %v", err)
	}

	llm, err := ollama.New(
		ollama.WithModel(model),
		ollama.WithServerURL(host),
	)
	if err != nil {
		log.Fatalf("ollama init: %v", err)
	}

	utils := append(mcpTools, tools.Calculator{}, DateTime{}, Shell{})
	opts := []agents.Option{agents.WithMaxIterations(maxIter)}
	if verbose {
		opts = append(opts, agents.WithCallbacksHandler(callbacks.LogHandler{}))
	}

	agent := agents.NewOneShotAgent(llm, utils, opts...)
	executor := agents.NewExecutor(agent)
	if query != "" {
		prompt(ctx, executor, query)
		return
	}

	scanner := bufio.NewScanner(os.Stdin)
	for {
		fmt.Print("You: ")
		if !scanner.Scan() {
			break
		}

		line := strings.TrimSpace(scanner.Text())
		if line == "" {
			continue
		}

		if line == "exit" || line == "quit" {
			break
		}

		prompt(ctx, executor, line)
		fmt.Println()
	}
}

func prompt(ctx context.Context, executor *agents.Executor, prompt string) {
	result, err := executor.Call(ctx, map[string]any{"input": prompt})
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		return
	}

	if output, ok := result["output"]; ok {
		fmt.Printf("Agent> %s\n", output)
	}
}
