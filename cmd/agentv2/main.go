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
	"github.com/tmc/langchaingo/memory"
	"github.com/tmc/langchaingo/tools"
)

// ---------- local tools ----------

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
	return `Execute a shell command. Input: the raw command.
Examples: ls -la, mkdir project, cd project && go build .
NEVER use this tool to create or write files. Use write_file instead.`
}
func (Shell) Call(_ context.Context, input string) (string, error) {
	input = strings.TrimSpace(input)
	input = strings.Trim(input, "`")
	input = strings.TrimSpace(input)

	// reject commands that try to write files via echo/cat/printf
	lower := strings.ToLower(input)
	if (strings.Contains(lower, "echo") || strings.Contains(lower, "printf") || strings.Contains(lower, "cat <<")) &&
		strings.ContainsAny(input, ">") {
		return "", fmt.Errorf("use write_file tool to create files, not shell")
	}

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

type ReadFile struct{}

func (ReadFile) Name() string { return "read_file" }
func (ReadFile) Description() string {
	return "Read contents of a file. Input: file path, e.g. /etc/hosts"
}
func (ReadFile) Call(_ context.Context, input string) (string, error) {
	path := strings.TrimSpace(input)
	path = strings.Trim(path, "`\"'")
	data, err := os.ReadFile(path)
	if err != nil {
		return "", fmt.Errorf("read_file: %w", err)
	}
	s := string(data)
	if len(s) > 8000 {
		s = s[:8000] + "\n... (truncated)"
	}
	return s, nil
}

type WriteFile struct{}

func (WriteFile) Name() string { return "write_file" }
func (WriteFile) Description() string {
	return `Create or overwrite a file. ALWAYS use this tool to write files, never use shell.
Input format: filepath|content
Example: project/main.go|package main

import "fmt"

func main() {
	fmt.Println("Hello, world!")
}
Everything after the first | is the file content, including newlines.`
}
func (WriteFile) Call(_ context.Context, input string) (string, error) {
	input = strings.TrimSpace(input)
	parts := strings.SplitN(input, "|", 2)
	if len(parts) != 2 {
		return "", fmt.Errorf("write_file: expected 'path|content', got: %s", input)
	}
	path := strings.TrimSpace(parts[0])
	content := parts[1]
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return "", fmt.Errorf("write_file: %w", err)
	}
	return fmt.Sprintf("wrote %d bytes to %s", len(content), path), nil
}

// ---------- prompt ----------

const systemPrefix = `You are an expert AI agent that solves complex tasks step by step.
You have access to tools and MUST use them to gather information and take actions.

CRITICAL RULES:
1. ALWAYS make a plan before acting. Think about ALL steps needed.
2. Execute ONE tool at a time, observe the result, then decide the next step.
3. Do NOT give a Final Answer until you have completed ALL steps of your plan.
4. If a step fails, try an alternative approach.
5. For complex tasks, use multiple tools in sequence.
6. NEVER write code blocks or markdown. ALWAYS use tools.
7. When writing files, use the write_file tool, not shell echo.

You have access to these tools:

{{.tool_descriptions}}

`

const formatInstructions = `ALWAYS respond in this EXACT format:

Question: the task to complete
Thought: plan your approach, what steps are needed
Action: tool name (one of [{{.tool_names}}])
Action Input: input for the tool
Observation: the tool result (you will receive this)

... repeat Thought/Action/Action Input/Observation for EACH step ...

Thought: I have completed all steps and have the final result
Final Answer: summary of what was done and the results

IMPORTANT:
- You MUST include Action Input even if the tool ignores it.
- NEVER skip steps. Complete the ENTIRE task before Final Answer.
- If your task has multiple parts, handle ALL of them.`

const suffixTemplate = `Question: {{.input}}
{{.agent_scratchpad}}`

const defaultQuery = "create a directory called project, inside it create main.go with a hello world program, then compile and run it. You: what time is it, what files are in current directory, and how much disk space is left. You: find all .go files in current directory, count lines in each, tell me which is longest"

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
	flag.StringVar(&mcpURL, "mcp", "", "MCP server URL (optional, e.g. http://localhost:9090/mcp)")
	flag.IntVar(&maxIter, "max-iter", 50, "max agent reasoning iterations")
	flag.StringVar(&query, "query", defaultQuery, "single query (skip for interactive REPL)")
	flag.BoolVar(&verbose, "verbose", false, "print agent reasoning steps")
	flag.Parse()

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt)
	defer cancel()

	// --- 1. Collect tools ---
	var allTools []tools.Tool

	// MCP tools (optional)
	if mcpURL != "" {
		mcp, err := mcpclient.NewStreamableHttpClient(mcpURL)
		if err != nil {
			log.Fatalf("mcp client: %v", err)
		}
		defer mcp.Close()

		a, err := adapter.New(mcp)
		if err != nil {
			log.Fatalf("mcp adapter: %v", err)
		}
		mcpTools, err := a.Tools()
		if err != nil {
			log.Fatalf("mcp tools: %v", err)
		}
		allTools = append(allTools, mcpTools...)
		log.Printf("loaded %d MCP tools from %s", len(mcpTools), mcpURL)
	}

	allTools = append(allTools,
		tools.Calculator{},
		DateTime{},
		Shell{},
		ReadFile{},
		WriteFile{},
	)

	log.Printf("total tools: %d", len(allTools))
	for _, t := range allTools {
		log.Printf("  - %s", t.Name())
	}

	// --- 2. Ollama ---
	llm, err := ollama.New(
		ollama.WithModel(model),
		ollama.WithServerURL(host),
	)
	if err != nil {
		log.Fatalf("ollama: %v", err)
	}

	// --- 3. Memory (persists across REPL turns) ---
	mem := memory.NewConversationBuffer()

	// --- 4. Agent with custom prompt ---
	opts := []agents.Option{
		agents.WithMaxIterations(maxIter),
		agents.WithPromptPrefix(systemPrefix),
		agents.WithPromptFormatInstructions(formatInstructions),
		agents.WithPromptSuffix(suffixTemplate),
	}
	if verbose {
		opts = append(opts, agents.WithCallbacksHandler(callbacks.LogHandler{}))
	}

	agent := agents.NewOneShotAgent(llm, allTools, opts...)
	executor := agents.NewExecutor(agent, agents.WithMemory(mem))

	// --- 5. Run ---
	if query != "" {
		runQuery(ctx, executor, query)
		return
	}

	toolNames := make([]string, len(allTools))
	for i, t := range allTools {
		toolNames[i] = t.Name()
	}

	fmt.Println("╔══════════════════════════════════════════════╗")
	fmt.Printf("║  Agent  |  model: %-26s ║\n", model)
	fmt.Printf("║  tools: %-37s ║\n", strings.Join(toolNames, ", "))
	fmt.Println("║  type 'exit' to quit                         ║")
	fmt.Println("╚══════════════════════════════════════════════╝")
	fmt.Println()

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
		runQuery(ctx, executor, line)
		fmt.Println()
	}
}

func runQuery(ctx context.Context, executor *agents.Executor, query string) {
	result, err := executor.Call(ctx, map[string]any{"input": query})
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		return
	}
	if output, ok := result["output"]; ok {
		fmt.Printf("Agent> %s\n", output)
	}
}
