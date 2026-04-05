package main

import (
	"context"
	"fmt"
	"log"
	"net/http"
	"os/exec"
	"strings"

	"github.com/modelcontextprotocol/go-sdk/mcp"
)

type GetPodsInput struct {
	Namespace string `json:"namespace" jsonschema:"Kubernetes namespace, default: default"`
}

type CommandOutput struct {
	Result string `json:"result" jsonschema:"command output"`
}

func handleGetPods(ctx context.Context, _ *mcp.CallToolRequest, in GetPodsInput) (*mcp.CallToolResult, CommandOutput, error) {
	ns := in.Namespace
	if ns == "" {
		ns = "default"
	}

	return run(ctx, "kubectl", "get", "pods", "-n", ns, "-o", "wide")
}

func run(ctx context.Context, name string, args ...string) (*mcp.CallToolResult, CommandOutput, error) {
	cmd := exec.CommandContext(ctx, name, args...)
	out, err := cmd.CombinedOutput()
	text := strings.TrimSpace(string(out))

	if err != nil {
		msg := fmt.Sprintf("command failed: %v\n%s", err, text)
		return &mcp.CallToolResult{
			IsError: true,
			Content: []mcp.Content{&mcp.TextContent{Text: msg}},
		}, CommandOutput{}, nil
	}

	if len(text) > 8000 {
		text = text[:8000] + "\n... (truncated)"
	}

	return nil, CommandOutput{Result: text}, nil
}

func main() {
	impl := &mcp.Implementation{Name: "k8s-tools", Version: "v1.0.0"}
	s := mcp.NewServer(impl, nil)

	mcp.AddTool(s, &mcp.Tool{Name: "get_pods", Description: "List pods with status, restarts, age"}, handleGetPods)

	fn := func(*http.Request) *mcp.Server { return s }
	handler := mcp.NewStreamableHTTPHandler(fn, nil)

	http.Handle("/mcp", handler)
	log.Println("MCP server (official SDK) on :9090/mcp")
	log.Fatal(http.ListenAndServe(":9090", nil))
}
