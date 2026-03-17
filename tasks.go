package llmbench

import (
	"sort"
	"strings"
)

// BenchmarkTasks returns the full benchmark suite: 20×L1 + 20×L2 + 20×L3 = 60 tasks.
//
// Document ordering within each task simulates ranked retrieval and directly
// affects RAG quality metrics (MRR, NDCG@K). Noise documents (relevance=0)
// test CHR and penalize NDCG. Document positions are varied across tasks
// to enable Lost-in-the-Middle vulnerability analysis:
//   - Some tasks place noise first (tests attention under distraction)
//   - Some tasks embed noise between relevant documents (middle position)
//   - Some tasks place noise last (baseline)
func BenchmarkTasks() []Task {
	n1 := RAGDocument{ID: "noise-monitoring", Content: NoiseManifestUnrelated, Relevance: 0}
	n2 := RAGDocument{ID: "noise-namespace", Content: NoiseManifestNamespace, Relevance: 0}

	return []Task{
		// =====================================================================
		// L1: DIAGNOSTIC (20 tasks) — model must identify the problem
		// =====================================================================
		{
			ID: "L1-diag-001", Level: LevelDiagnostic,
			Description: "Identify the problem with the Pod in the default namespace.",
			Documents:   []RAGDocument{{ID: "pod-nginx-crashloop", Content: ManifestPodNginxCrashLoop, Relevance: 3}, n1},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"crashloopbackoff", "crash loop", "crashloop"}},
				ActionTerms:         []string{"kubectl logs", "kubectl describe", "logs", "describe"},
				ContextEntities:     map[string]string{"pod_name": "nginx", "namespace": "default", "state": "CrashLoopBackOff"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"get_pod_status", "get_pod_logs"},
			},
		},
		{
			ID: "L1-diag-002", Level: LevelDiagnostic,
			Description: "Identify why the nginx-worker Pod in production keeps restarting.",
			Documents:   []RAGDocument{n1, {ID: "pod-nginx-oom", Content: ManifestPodNginxOOM, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"oomkilled", "oom", "out of memory"}},
				ActionTerms:         []string{"kubectl describe", "memory", "limit", "resources"},
				ContextEntities:     map[string]string{"pod_name": "nginx-worker", "namespace": "production", "exit_code": "137"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"get_pod_status", "describe_pod"},
			},
		},
		{
			ID: "L1-diag-003", Level: LevelDiagnostic,
			Description: "Identify why the api-server Pod in staging is not starting.",
			Documents:   []RAGDocument{{ID: "pod-imagepull", Content: ManifestPodImagePullError, Relevance: 3}, n1},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"imagepullbackoff", "image pull", "imagepull", "pull image"}},
				ActionTerms:         []string{"image", "tag", "kubectl describe", "correct"},
				ContextEntities:     map[string]string{"pod_name": "api-server", "namespace": "staging", "image_tag": "v2.1-typo"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"get_pod_status", "describe_pod"},
			},
		},
		{
			ID: "L1-diag-004", Level: LevelDiagnostic,
			Description: "Identify why the ml-trainer Pod cannot start.",
			Documents:   []RAGDocument{{ID: "pod-pending", Content: ManifestPodPending, Relevance: 3}, n2},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"pending", "unschedulable", "schedule"}},
				ActionTerms:         []string{"describe", "node", "resource", "memory"},
				ContextEntities:     map[string]string{"pod_name": "ml-trainer", "namespace": "ml-jobs", "memory": "16Gi"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_nodes"},
			},
		},
		{
			ID: "L1-diag-005", Level: LevelDiagnostic,
			Description: "Identify why the PVC data-pvc is stuck in Pending state.",
			Documents:   []RAGDocument{n1, {ID: "pvc-pending", Content: ManifestPVCPending, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"storageclass", "fast-ssd", "not found", "provisioning"}},
				ActionTerms:         []string{"storageclass", "kubectl describe", "kubectl get sc"},
				ContextEntities:     map[string]string{"pvc_name": "data-pvc", "namespace": "ml-jobs", "storage_class": "fast-ssd"},
				ForbiddenPatterns:   []string{"delete namespace", "delete pvc"},
				OptimalToolSequence: []string{"describe_pvc", "get_storageclasses"},
			},
		},
		{
			ID: "L1-diag-006", Level: LevelDiagnostic,
			Description: "Identify why the app-service-account cannot read pods.",
			Documents:   []RAGDocument{{ID: "rbac-missing", Content: ManifestRBACMissing, Relevance: 3}, n2},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"rolebinding", "binding", "rbac", "permission"}},
				ActionTerms:         []string{"rolebinding", "kubectl create", "kubectl apply", "bind"},
				ContextEntities:     map[string]string{"account": "app-service-account", "role": "pod-reader", "namespace": "production"},
				ForbiddenPatterns:   []string{"delete namespace", "delete role"},
				OptimalToolSequence: []string{"get_rolebindings", "create_rolebinding"},
			},
		},
		{
			ID: "L1-diag-007", Level: LevelDiagnostic,
			Description: "Identify why the payment-api Pod keeps restarting despite running.",
			Documents:   []RAGDocument{{ID: "pod-probe-fail", Content: ManifestPodProbeFailure, Relevance: 3}, n1},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"liveness", "probe", "health", "503"}},
				ActionTerms:         []string{"probe", "healthz", "kubectl describe", "logs"},
				ContextEntities:     map[string]string{"pod_name": "payment-api", "namespace": "production", "probe_path": "/healthz"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_pod_logs"},
			},
		},
		{
			ID: "L1-diag-008", Level: LevelDiagnostic,
			Description: "Identify why the data-processor Pod exits immediately after starting.",
			Documents:   []RAGDocument{n2, {ID: "pod-wrong-cmd", Content: ManifestPodWrongCommand, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"not found", "executable", "command", "127"}},
				ActionTerms:         []string{"command", "entrypoint", "kubectl describe", "kubectl logs"},
				ContextEntities:     map[string]string{"pod_name": "data-processor", "namespace": "batch", "exit_code": "127"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_pod_logs"},
			},
		},
		{
			ID: "L1-diag-009", Level: LevelDiagnostic,
			Description: "Identify why the auth-service Pod is stuck in Pending.",
			Documents:   []RAGDocument{{ID: "pod-missing-secret", Content: ManifestPodMissingSecretRef, Relevance: 3}, n1},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"secret", "jwt-signing-key", "not found", "createcontainerconfigerror"}},
				ActionTerms:         []string{"secret", "kubectl create secret", "kubectl describe"},
				ContextEntities:     map[string]string{"pod_name": "auth-service", "namespace": "production", "secret_name": "jwt-signing-key"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "create_secret"},
			},
		},
		{
			ID: "L1-diag-010", Level: LevelDiagnostic,
			Description: "Identify why the gpu-inference Pod cannot be scheduled.",
			Documents:   []RAGDocument{n1, {ID: "pod-affinity", Content: ManifestPodNodeAffinity, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"affinity", "node", "selector", "nvidia-a100", "unschedulable"}},
				ActionTerms:         []string{"node", "label", "affinity", "kubectl describe", "kubectl get nodes"},
				ContextEntities:     map[string]string{"pod_name": "gpu-inference", "namespace": "ml-jobs", "label": "nvidia-a100"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_nodes"},
			},
		},
		{
			ID: "L1-diag-011", Level: LevelDiagnostic,
			Description: "Identify what happened to the log-aggregator Pod.",
			Documents:   []RAGDocument{{ID: "pod-evicted", Content: ManifestPodEvicted, Relevance: 3}, n2},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"evict", "ephemeral-storage", "disk", "low on resource"}},
				ActionTerms:         []string{"describe", "storage", "node", "kubectl describe"},
				ContextEntities:     map[string]string{"pod_name": "log-aggregator", "namespace": "monitoring", "reason": "Evicted"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_node_status"},
			},
		},
		{
			ID: "L1-diag-012", Level: LevelDiagnostic,
			Description: "Identify why the web-app Pod is stuck in PodInitializing.",
			Documents:   []RAGDocument{n1, {ID: "pod-init-fail", Content: ManifestPodInitContainerFail, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"init", "container", "db-service", "connection refused"}},
				ActionTerms:         []string{"init container", "kubectl logs", "kubectl describe", "db-service"},
				ContextEntities:     map[string]string{"pod_name": "web-app", "namespace": "default", "init_target": "db-service"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_pod_logs"},
			},
		},
		{
			ID: "L1-diag-013", Level: LevelDiagnostic,
			Description: "Identify why the nginx-deployment has no running Pods.",
			Documents:   []RAGDocument{{ID: "deploy-nginx", Content: ManifestDeploymentNginx, Relevance: 3}, n1},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"replicas", "0", "unavailable", "scaled"}},
				ActionTerms:         []string{"kubectl scale", "kubectl describe", "replicas"},
				ContextEntities:     map[string]string{"deployment": "nginx-deployment", "namespace": "default", "replicas": "0"},
				ForbiddenPatterns:   []string{"delete namespace", "delete deployment"},
				OptimalToolSequence: []string{"describe_deployment", "get_pods"},
			},
		},
		{
			ID: "L1-diag-014", Level: LevelDiagnostic,
			Description: "Identify why the frontend Deployment rollout is stuck.",
			Documents:   []RAGDocument{n2, {ID: "deploy-stuck", Content: ManifestDeploymentRolloutStuck, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"progressdeadlineexceeded", "progress deadline", "timed out", "rollout"}},
				ActionTerms:         []string{"kubectl rollout", "kubectl describe", "rollout undo", "image"},
				ContextEntities:     map[string]string{"deployment": "frontend", "namespace": "production", "image": "5.0-broken"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_deployment", "rollout_undo"},
			},
		},
		{
			ID: "L1-diag-015", Level: LevelDiagnostic,
			Description: "Identify why the nginx-service has no endpoints.",
			Documents:   []RAGDocument{{ID: "svc-nginx", Content: ManifestServiceNginx, Relevance: 3}, n1},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"selector", "nginx-typo", "mismatch", "label", "endpoints"}},
				ActionTerms:         []string{"kubectl describe", "kubectl get endpoints", "selector", "labels"},
				ContextEntities:     map[string]string{"service": "nginx-service", "namespace": "default", "wrong_selector": "nginx-typo"},
				ForbiddenPatterns:   []string{"delete namespace", "delete service"},
				OptimalToolSequence: []string{"describe_service", "get_endpoints"},
			},
		},
		{
			ID: "L1-diag-016", Level: LevelDiagnostic,
			Description: "Identify why the api-gateway Service is not forwarding traffic correctly.",
			Documents:   []RAGDocument{n2, {ID: "svc-wrong-port", Content: ManifestServiceWrongTargetPort, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"targetport", "9090", "port", "mismatch"}},
				ActionTerms:         []string{"kubectl describe", "kubectl edit", "targetPort", "port"},
				ContextEntities:     map[string]string{"service": "api-gateway", "namespace": "production", "target_port": "9090"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_service", "get_pods"},
			},
		},
		{
			ID: "L1-diag-017", Level: LevelDiagnostic,
			Description: "Identify why the db-migration Job has failed.",
			Documents:   []RAGDocument{{ID: "job-failed", Content: ManifestJobFailed, Relevance: 3}, n1},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"backofflimitexceeded", "backoff limit", "failed"}},
				ActionTerms:         []string{"kubectl describe", "kubectl logs", "job"},
				ContextEntities:     map[string]string{"job_name": "db-migration", "namespace": "production", "reason": "BackoffLimitExceeded"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_job", "get_pod_logs"},
			},
		},
		{
			ID: "L1-diag-018", Level: LevelDiagnostic,
			Description: "Identify why the api-server Pods cannot receive incoming traffic.",
			Documents:   []RAGDocument{n1, {ID: "netpol-deny", Content: ManifestNetworkPolicyDenyAll, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"networkpolicy", "network policy", "deny", "ingress"}},
				ActionTerms:         []string{"networkpolicy", "kubectl describe", "kubectl get netpol", "ingress"},
				ContextEntities:     map[string]string{"policy": "deny-all-ingress", "namespace": "production", "target": "api-server"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"get_networkpolicies", "describe_networkpolicy"},
			},
		},
		{
			ID: "L1-diag-019", Level: LevelDiagnostic,
			Description: "Identify why the Ingress app-ingress returns 503 errors.",
			Documents:   []RAGDocument{{ID: "ingress-wrong", Content: ManifestIngressWrongBackend, Relevance: 3}, n2},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"api-gateway-old", "backend", "service", "not found"}},
				ActionTerms:         []string{"kubectl describe", "kubectl edit", "ingress", "backend"},
				ContextEntities:     map[string]string{"ingress": "app-ingress", "namespace": "production", "wrong_backend": "api-gateway-old"},
				ForbiddenPatterns:   []string{"delete namespace", "delete ingress"},
				OptimalToolSequence: []string{"describe_ingress", "get_services"},
			},
		},
		{
			ID: "L1-diag-020", Level: LevelDiagnostic,
			Description: "Identify why the HPA api-hpa is not scaling the deployment.",
			Documents:   []RAGDocument{n1, {ID: "hpa-wrong", Content: ManifestHPAConfig, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"api-v1", "not found", "scaletargetref", "failedgetscale"}},
				ActionTerms:         []string{"kubectl describe", "kubectl edit", "hpa", "scaleTargetRef"},
				ContextEntities:     map[string]string{"hpa": "api-hpa", "namespace": "production", "wrong_target": "api-v1"},
				ForbiddenPatterns:   []string{"delete namespace", "delete hpa"},
				OptimalToolSequence: []string{"describe_hpa", "get_deployments"},
			},
		},

		// =====================================================================
		// L2: REPAIR (20 tasks) — model must diagnose AND propose a concrete fix
		// =====================================================================
		{
			ID: "L2-repair-001", Level: LevelRepair,
			Description: "The nginx Pod in default namespace is unhealthy. Diagnose and fix the issue.",
			Documents:   []RAGDocument{{ID: "pod-nginx-crashloop", Content: ManifestPodNginxCrashLoop, Relevance: 3}, {ID: "deploy-nginx", Content: ManifestDeploymentNginx, Relevance: 1}, n1},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"crashloopbackoff", "crash loop", "crashloop"}, {"restart", "exit", "failed", "error"}},
				ActionTerms:         []string{"kubectl logs", "kubectl describe", "kubectl delete pod", "kubectl rollout"},
				ContextEntities:     map[string]string{"pod_name": "nginx", "namespace": "default", "state": "CrashLoopBackOff"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"get_pod_status", "get_pod_logs", "delete_pod"},
			},
		},
		{
			ID: "L2-repair-002", Level: LevelRepair,
			Description: "The nginx-worker Pod is being OOM-killed repeatedly. Diagnose and fix.",
			Documents:   []RAGDocument{n1, {ID: "pod-nginx-oom", Content: ManifestPodNginxOOM, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"oomkilled", "oom", "out of memory"}, {"memory", "limit", "resource"}},
				ActionTerms:         []string{"kubectl edit", "kubectl patch", "increase", "limit", "memory"},
				ContextEntities:     map[string]string{"pod_name": "nginx-worker", "namespace": "production", "memory_limit": "64Mi"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "patch_deployment"},
			},
		},
		{
			ID: "L2-repair-003", Level: LevelRepair,
			Description: "The ml-trainer Pod cannot be scheduled. Diagnose and fix.",
			Documents:   []RAGDocument{{ID: "pod-pending", Content: ManifestPodPending, Relevance: 3}, n1, {ID: "node-status", Content: ManifestNodeStatusFull, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"pending", "unschedulable", "schedule"}, {"memory", "insufficient", "resource"}},
				ActionTerms:         []string{"reduce", "request", "node", "scale", "resource"},
				ContextEntities:     map[string]string{"pod_name": "ml-trainer", "namespace": "ml-jobs", "memory": "16Gi"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_node_status", "patch_pod_resources"},
			},
		},
		{
			ID: "L2-repair-004", Level: LevelRepair,
			Description: "The api-server Pod cannot pull its container image. Fix the issue.",
			Documents:   []RAGDocument{{ID: "pod-imagepull", Content: ManifestPodImagePullError, Relevance: 3}, n2},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"imagepullbackoff", "image pull", "imagepull"}, {"v2.1-typo", "tag", "image"}},
				ActionTerms:         []string{"kubectl edit", "kubectl set image", "kubectl patch", "image"},
				ContextEntities:     map[string]string{"pod_name": "api-server", "namespace": "staging", "image": "v2.1-typo"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "patch_pod_image"},
			},
		},
		{
			ID: "L2-repair-005", Level: LevelRepair,
			Description: "The data-pvc PersistentVolumeClaim is stuck. Diagnose and fix.",
			Documents:   []RAGDocument{n1, {ID: "pvc-pending", Content: ManifestPVCPending, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"storageclass", "fast-ssd", "not found"}, {"provision", "storage"}},
				ActionTerms:         []string{"storageclass", "kubectl create", "kubectl apply", "kubectl patch"},
				ContextEntities:     map[string]string{"pvc_name": "data-pvc", "namespace": "ml-jobs", "storage_class": "fast-ssd"},
				ForbiddenPatterns:   []string{"delete namespace", "delete pvc"},
				OptimalToolSequence: []string{"describe_pvc", "create_storageclass"},
			},
		},
		{
			ID: "L2-repair-006", Level: LevelRepair,
			Description: "The app-service-account lacks permissions to read pods. Fix the RBAC setup.",
			Documents:   []RAGDocument{{ID: "rbac-missing", Content: ManifestRBACMissing, Relevance: 3}, n1},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"rolebinding", "binding", "rbac"}, {"app-service-account", "pod-reader"}},
				ActionTerms:         []string{"kubectl create rolebinding", "kubectl apply", "rolebinding"},
				ContextEntities:     map[string]string{"account": "app-service-account", "role": "pod-reader", "namespace": "production"},
				ForbiddenPatterns:   []string{"delete namespace", "delete role", "delete serviceaccount"},
				OptimalToolSequence: []string{"get_rolebindings", "create_rolebinding"},
			},
		},
		{
			ID: "L2-repair-007", Level: LevelRepair,
			Description: "The payment-api Pod is restarting due to failing health checks. Fix it.",
			Documents:   []RAGDocument{n2, {ID: "pod-probe-fail", Content: ManifestPodProbeFailure, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"liveness", "probe", "503"}, {"health", "restart", "fail"}},
				ActionTerms:         []string{"probe", "initialDelaySeconds", "kubectl edit", "kubectl patch"},
				ContextEntities:     map[string]string{"pod_name": "payment-api", "namespace": "production", "probe_path": "/healthz"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_pod_logs", "patch_probe"},
			},
		},
		{
			ID: "L2-repair-008", Level: LevelRepair,
			Description: "The data-processor Pod crashes with exit code 127. Fix the issue.",
			Documents:   []RAGDocument{{ID: "pod-wrong-cmd", Content: ManifestPodWrongCommand, Relevance: 3}, n1},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"not found", "executable", "command", "127"}, {"process-data.sh", "entrypoint"}},
				ActionTerms:         []string{"command", "kubectl edit", "kubectl patch", "python"},
				ContextEntities:     map[string]string{"pod_name": "data-processor", "namespace": "batch", "command": "process-data.sh"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "patch_pod_command"},
			},
		},
		{
			ID: "L2-repair-009", Level: LevelRepair,
			Description: "The auth-service Pod cannot start because of a missing Secret. Fix it.",
			Documents:   []RAGDocument{n1, {ID: "pod-missing-secret", Content: ManifestPodMissingSecretRef, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"secret", "jwt-signing-key", "not found"}, {"createcontainerconfigerror", "config"}},
				ActionTerms:         []string{"kubectl create secret", "kubectl apply", "secret"},
				ContextEntities:     map[string]string{"pod_name": "auth-service", "namespace": "production", "secret_name": "jwt-signing-key"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "create_secret"},
			},
		},
		{
			ID: "L2-repair-010", Level: LevelRepair,
			Description: "The gpu-inference Pod is stuck Pending due to node affinity. Fix scheduling.",
			Documents:   []RAGDocument{{ID: "pod-affinity", Content: ManifestPodNodeAffinity, Relevance: 3}, n2, {ID: "node-status", Content: ManifestNodeStatusFull, Relevance: 1}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"affinity", "node", "nvidia-a100"}, {"unschedulable", "selector", "label"}},
				ActionTerms:         []string{"kubectl label node", "kubectl edit", "affinity", "node"},
				ContextEntities:     map[string]string{"pod_name": "gpu-inference", "namespace": "ml-jobs", "label": "nvidia-a100"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_nodes", "label_node"},
			},
		},
		{
			ID: "L2-repair-011", Level: LevelRepair,
			Description: "The log-aggregator Pod was evicted. Diagnose and prevent recurrence.",
			Documents:   []RAGDocument{n1, {ID: "pod-evicted", Content: ManifestPodEvicted, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"evict", "ephemeral-storage", "disk"}, {"low on resource", "threshold"}},
				ActionTerms:         []string{"emptyDir", "sizeLimit", "kubectl edit", "ephemeral", "storage"},
				ContextEntities:     map[string]string{"pod_name": "log-aggregator", "namespace": "monitoring", "reason": "Evicted"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_node_status", "patch_pod_storage"},
			},
		},
		{
			ID: "L2-repair-012", Level: LevelRepair,
			Description: "The web-app Pod is stuck in PodInitializing. Fix the init container issue.",
			Documents:   []RAGDocument{{ID: "pod-init-fail", Content: ManifestPodInitContainerFail, Relevance: 3}, n2},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"init", "container", "db-service"}, {"connection refused", "5432"}},
				ActionTerms:         []string{"kubectl logs", "service", "db-service", "kubectl create"},
				ContextEntities:     map[string]string{"pod_name": "web-app", "namespace": "default", "init_target": "db-service"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_services", "create_service"},
			},
		},
		{
			ID: "L2-repair-013", Level: LevelRepair,
			Description: "The nginx-deployment has 0 running replicas. Fix the deployment.",
			Documents:   []RAGDocument{n1, {ID: "deploy-nginx", Content: ManifestDeploymentNginx, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"replicas", "0", "unavailable"}, {"scaled", "minimum"}},
				ActionTerms:         []string{"kubectl scale", "kubectl patch", "replicas"},
				ContextEntities:     map[string]string{"deployment": "nginx-deployment", "namespace": "default", "replicas": "0"},
				ForbiddenPatterns:   []string{"delete namespace", "delete deployment"},
				OptimalToolSequence: []string{"describe_deployment", "scale_deployment"},
			},
		},
		{
			ID: "L2-repair-014", Level: LevelRepair,
			Description: "The frontend Deployment rollout timed out. Roll back or fix.",
			Documents:   []RAGDocument{{ID: "deploy-stuck", Content: ManifestDeploymentRolloutStuck, Relevance: 3}, n1},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"progressdeadlineexceeded", "progress deadline", "timed out"}, {"5.0-broken", "image", "rollout"}},
				ActionTerms:         []string{"kubectl rollout undo", "kubectl set image", "rollback"},
				ContextEntities:     map[string]string{"deployment": "frontend", "namespace": "production", "image": "5.0-broken"},
				ForbiddenPatterns:   []string{"delete namespace", "delete deployment"},
				OptimalToolSequence: []string{"describe_deployment", "rollout_undo"},
			},
		},
		{
			ID: "L2-repair-015", Level: LevelRepair,
			Description: "The nginx-service has no endpoints and traffic is failing. Fix it.",
			Documents:   []RAGDocument{{ID: "svc-nginx", Content: ManifestServiceNginx, Relevance: 3}, n2},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"selector", "nginx-typo", "mismatch"}, {"label", "endpoints"}},
				ActionTerms:         []string{"kubectl patch", "kubectl edit", "selector", "nginx"},
				ContextEntities:     map[string]string{"service": "nginx-service", "namespace": "default", "wrong_selector": "nginx-typo"},
				ForbiddenPatterns:   []string{"delete namespace", "delete service"},
				OptimalToolSequence: []string{"describe_service", "patch_service_selector"},
			},
		},
		{
			ID: "L2-repair-016", Level: LevelRepair,
			Description: "The api-gateway Service forwards to the wrong port. Fix the port mapping.",
			Documents:   []RAGDocument{n1, {ID: "svc-wrong-port", Content: ManifestServiceWrongTargetPort, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"targetport", "9090", "port"}, {"mismatch", "wrong"}},
				ActionTerms:         []string{"kubectl edit", "kubectl patch", "targetPort"},
				ContextEntities:     map[string]string{"service": "api-gateway", "namespace": "production", "target_port": "9090"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_service", "patch_service_port"},
			},
		},
		{
			ID: "L2-repair-017", Level: LevelRepair,
			Description: "The db-migration Job has failed. Diagnose, fix and re-run.",
			Documents:   []RAGDocument{{ID: "job-failed", Content: ManifestJobFailed, Relevance: 3}, n2},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"backofflimitexceeded", "backoff limit"}, {"failed", "job"}},
				ActionTerms:         []string{"kubectl delete job", "kubectl create", "kubectl logs", "rerun"},
				ContextEntities:     map[string]string{"job_name": "db-migration", "namespace": "production", "reason": "BackoffLimitExceeded"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_job", "get_pod_logs", "delete_job", "create_job"},
			},
		},
		{
			ID: "L2-repair-018", Level: LevelRepair,
			Description: "The NetworkPolicy is blocking all traffic to api-server. Fix connectivity.",
			Documents:   []RAGDocument{n1, {ID: "netpol-deny", Content: ManifestNetworkPolicyDenyAll, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"networkpolicy", "network policy", "deny"}, {"ingress", "block"}},
				ActionTerms:         []string{"kubectl edit", "kubectl delete networkpolicy", "kubectl patch", "ingress"},
				ContextEntities:     map[string]string{"policy": "deny-all-ingress", "namespace": "production", "target": "api-server"},
				ForbiddenPatterns:   []string{"delete namespace"},
				OptimalToolSequence: []string{"describe_networkpolicy", "patch_networkpolicy"},
			},
		},
		{
			ID: "L2-repair-019", Level: LevelRepair,
			Description: "The Ingress app-ingress points to a non-existent backend. Fix routing.",
			Documents:   []RAGDocument{{ID: "ingress-wrong", Content: ManifestIngressWrongBackend, Relevance: 3}, n1},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"api-gateway-old", "backend"}, {"service", "not found", "ingress"}},
				ActionTerms:         []string{"kubectl edit", "kubectl patch", "backend", "service"},
				ContextEntities:     map[string]string{"ingress": "app-ingress", "namespace": "production", "wrong_backend": "api-gateway-old"},
				ForbiddenPatterns:   []string{"delete namespace", "delete ingress"},
				OptimalToolSequence: []string{"describe_ingress", "patch_ingress_backend"},
			},
		},
		{
			ID: "L2-repair-020", Level: LevelRepair,
			Description: "The HPA api-hpa cannot find its target Deployment. Fix the autoscaler.",
			Documents:   []RAGDocument{n2, {ID: "hpa-wrong", Content: ManifestHPAConfig, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"api-v1", "not found", "scaletargetref"}, {"failedgetscale", "hpa"}},
				ActionTerms:         []string{"kubectl edit", "kubectl patch", "scaleTargetRef", "api-server"},
				ContextEntities:     map[string]string{"hpa": "api-hpa", "namespace": "production", "wrong_target": "api-v1"},
				ForbiddenPatterns:   []string{"delete namespace", "delete hpa"},
				OptimalToolSequence: []string{"describe_hpa", "patch_hpa_target"},
			},
		},

		// =====================================================================
		// L3: MULTI-STEP (20 tasks) — cross-reference multiple resources
		// =====================================================================
		{
			ID: "L3-multi-001", Level: LevelMultiStep,
			Description: "The nginx service is not routing traffic. Multiple resources may be misconfigured. Find and fix ALL issues.",
			Documents:   []RAGDocument{{ID: "svc-nginx", Content: ManifestServiceNginx, Relevance: 3}, n1, {ID: "deploy-nginx", Content: ManifestDeploymentNginx, Relevance: 3}, {ID: "pod-nginx", Content: ManifestPodNginxCrashLoop, Relevance: 1}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"selector", "nginx-typo", "mismatch", "label"}, {"replicas", "scale", "unavailable"}},
				ActionTerms:         []string{"kubectl edit", "kubectl patch", "kubectl scale", "selector"},
				ContextEntities:     map[string]string{"service_name": "nginx-service", "wrong_selector": "nginx-typo", "deployment": "nginx-deployment"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"get_service", "get_endpoints", "patch_service_selector", "scale_deployment"},
			},
		},
		{
			ID: "L3-multi-002", Level: LevelMultiStep,
			Description: "The api-server Deployment has 0 ready replicas. The database connection is broken. Find and fix ALL issues.",
			Documents:   []RAGDocument{{ID: "deploy-api", Content: ManifestDeploymentAPIServer, Relevance: 3}, n1, {ID: "cm-db", Content: ManifestConfigMapDB, Relevance: 3}, {ID: "svc-pg", Content: ManifestServicePostgres, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"postgres-svc-old", "db_host", "db host", "environment", "env"}, {"postgres-service", "configmap", "db-config"}},
				ActionTerms:         []string{"kubectl edit", "kubectl patch", "kubectl set env", "env", "db_host"},
				ContextEntities:     map[string]string{"deployment": "api-server", "wrong_host": "postgres-svc-old", "correct_host": "postgres-service", "namespace": "production"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_deployment", "get_configmap", "get_service", "patch_deployment_env"},
			},
		},
		{
			ID: "L3-multi-003", Level: LevelMultiStep,
			Description: "The web-app cannot start and the database service is missing. Find and fix ALL issues.",
			Documents:   []RAGDocument{{ID: "pod-init-fail", Content: ManifestPodInitContainerFail, Relevance: 3}, n2, {ID: "svc-pg", Content: ManifestServicePostgres, Relevance: 1}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"init", "container", "db-service"}, {"connection refused", "5432", "service"}},
				ActionTerms:         []string{"kubectl create service", "kubectl apply", "db-service"},
				ContextEntities:     map[string]string{"pod_name": "web-app", "namespace": "default", "init_target": "db-service"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_services", "create_service"},
			},
		},
		{
			ID: "L3-multi-004", Level: LevelMultiStep,
			Description: "The payment-api is down. Health checks fail and the NetworkPolicy may block traffic. Find and fix ALL issues.",
			Documents:   []RAGDocument{{ID: "pod-probe", Content: ManifestPodProbeFailure, Relevance: 3}, n1, {ID: "netpol", Content: ManifestNetworkPolicyDenyAll, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"liveness", "probe", "503"}, {"networkpolicy", "deny", "ingress"}},
				ActionTerms:         []string{"kubectl edit", "kubectl patch", "probe", "networkpolicy"},
				ContextEntities:     map[string]string{"pod_name": "payment-api", "policy": "deny-all-ingress", "namespace": "production"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_networkpolicies", "patch_probe", "patch_networkpolicy"},
			},
		},
		{
			ID: "L3-multi-005", Level: LevelMultiStep,
			Description: "The frontend rollout is stuck and the Ingress returns errors. Find and fix ALL issues.",
			Documents:   []RAGDocument{{ID: "deploy-stuck", Content: ManifestDeploymentRolloutStuck, Relevance: 3}, n2, {ID: "ingress-wrong", Content: ManifestIngressWrongBackend, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"progressdeadlineexceeded", "timed out", "rollout"}, {"api-gateway-old", "backend", "ingress"}},
				ActionTerms:         []string{"kubectl rollout undo", "kubectl edit", "backend"},
				ContextEntities:     map[string]string{"deployment": "frontend", "ingress": "app-ingress", "wrong_backend": "api-gateway-old"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_deployment", "rollout_undo", "describe_ingress", "patch_ingress_backend"},
			},
		},
		{
			ID: "L3-multi-006", Level: LevelMultiStep,
			Description: "The HPA cannot scale the api-server and the Deployment env is wrong. Find and fix ALL issues.",
			Documents:   []RAGDocument{{ID: "hpa-wrong", Content: ManifestHPAConfig, Relevance: 3}, n1, {ID: "deploy-api", Content: ManifestDeploymentAPIServer, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"api-v1", "scaletargetref", "not found"}, {"postgres-svc-old", "db_host", "env"}},
				ActionTerms:         []string{"kubectl edit", "kubectl patch", "scaleTargetRef", "env"},
				ContextEntities:     map[string]string{"hpa": "api-hpa", "deployment": "api-server", "wrong_target": "api-v1", "wrong_host": "postgres-svc-old"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_hpa", "patch_hpa_target", "describe_deployment", "patch_deployment_env"},
			},
		},
		{
			ID: "L3-multi-007", Level: LevelMultiStep,
			Description: "The ml-trainer Pod is Pending and the PVC cannot be provisioned. Find and fix ALL issues.",
			Documents:   []RAGDocument{{ID: "pod-pending", Content: ManifestPodPending, Relevance: 3}, n2, {ID: "pvc-pending", Content: ManifestPVCPending, Relevance: 3}, {ID: "node-status", Content: ManifestNodeStatusFull, Relevance: 1}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"unschedulable", "pending", "memory"}, {"storageclass", "fast-ssd", "not found"}},
				ActionTerms:         []string{"resource", "storageclass", "kubectl create", "kubectl patch"},
				ContextEntities:     map[string]string{"pod_name": "ml-trainer", "pvc_name": "data-pvc", "storage_class": "fast-ssd"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "describe_pvc", "create_storageclass", "patch_pod_resources"},
			},
		},
		{
			ID: "L3-multi-008", Level: LevelMultiStep,
			Description: "The api-server has RBAC issues and the Service selector is wrong. Users cannot access the API. Fix ALL issues.",
			Documents:   []RAGDocument{{ID: "rbac-missing", Content: ManifestRBACMissing, Relevance: 3}, n1, {ID: "svc-nginx", Content: ManifestServiceNginx, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"rolebinding", "rbac", "permission"}, {"selector", "nginx-typo", "mismatch"}},
				ActionTerms:         []string{"rolebinding", "selector", "kubectl create", "kubectl patch"},
				ContextEntities:     map[string]string{"account": "app-service-account", "service": "nginx-service", "wrong_selector": "nginx-typo"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"get_rolebindings", "create_rolebinding", "patch_service_selector"},
			},
		},
		{
			ID: "L3-multi-009", Level: LevelMultiStep,
			Description: "The auth-service cannot start (missing Secret) and the Service has wrong targetPort. Fix ALL issues.",
			Documents:   []RAGDocument{{ID: "pod-secret", Content: ManifestPodMissingSecretRef, Relevance: 3}, n2, {ID: "svc-port", Content: ManifestServiceWrongTargetPort, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"secret", "jwt-signing-key", "not found"}, {"targetport", "9090", "port"}},
				ActionTerms:         []string{"kubectl create secret", "kubectl edit", "targetPort"},
				ContextEntities:     map[string]string{"pod_name": "auth-service", "secret_name": "jwt-signing-key", "service": "api-gateway", "target_port": "9090"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "create_secret", "describe_service", "patch_service_port"},
			},
		},
		{
			ID: "L3-multi-010", Level: LevelMultiStep,
			Description: "The data-processor crashes and the db-migration Job has failed. Batch pipeline is broken. Fix ALL issues.",
			Documents:   []RAGDocument{{ID: "pod-cmd", Content: ManifestPodWrongCommand, Relevance: 3}, n1, {ID: "job-failed", Content: ManifestJobFailed, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"command", "not found", "127"}, {"backofflimitexceeded", "job", "failed"}},
				ActionTerms:         []string{"command", "kubectl edit", "kubectl delete job", "kubectl create"},
				ContextEntities:     map[string]string{"pod_name": "data-processor", "job_name": "db-migration", "exit_code": "127"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "patch_pod_command", "describe_job", "recreate_job"},
			},
		},
		{
			ID: "L3-multi-011", Level: LevelMultiStep,
			Description: "The cache-client Pod has DNS errors and the gpu-inference Pod is unschedulable. Fix ALL issues.",
			Documents:   []RAGDocument{{ID: "pod-dns", Content: ManifestPodDNSError, Relevance: 3}, n2, {ID: "pod-affinity", Content: ManifestPodNodeAffinity, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"dns", "no such host", "redis-master"}, {"affinity", "nvidia-a100", "unschedulable"}},
				ActionTerms:         []string{"service", "kubectl create", "kubectl label node", "dns"},
				ContextEntities:     map[string]string{"pod_dns": "cache-client", "pod_gpu": "gpu-inference", "redis_host": "redis-master.cache.svc.cluster.local"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "create_service", "describe_pod", "label_node"},
			},
		},
		{
			ID: "L3-multi-012", Level: LevelMultiStep,
			Description: "The nginx-deployment has 0 replicas and the Service selector is wrong. No traffic flows. Fix ALL issues.",
			Documents:   []RAGDocument{n1, {ID: "deploy-nginx", Content: ManifestDeploymentNginx, Relevance: 3}, {ID: "svc-nginx", Content: ManifestServiceNginx, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"replicas", "0", "unavailable"}, {"selector", "nginx-typo", "mismatch"}},
				ActionTerms:         []string{"kubectl scale", "kubectl patch", "selector", "replicas"},
				ContextEntities:     map[string]string{"deployment": "nginx-deployment", "service": "nginx-service", "wrong_selector": "nginx-typo"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_deployment", "scale_deployment", "describe_service", "patch_service_selector"},
			},
		},
		{
			ID: "L3-multi-013", Level: LevelMultiStep,
			Description: "The log-aggregator was evicted and the worker-pool Deployment cannot scale due to quota. Fix ALL issues.",
			Documents:   []RAGDocument{{ID: "pod-evicted", Content: ManifestPodEvicted, Relevance: 3}, n1, {ID: "deploy-quota", Content: ManifestDeploymentQuotaExceeded, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"evict", "ephemeral-storage"}, {"quota", "exceeded", "forbidden"}},
				ActionTerms:         []string{"storage", "quota", "kubectl edit", "kubectl patch"},
				ContextEntities:     map[string]string{"pod_name": "log-aggregator", "deployment": "worker-pool", "reason": "Evicted"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "get_resourcequotas", "patch_quota"},
			},
		},
		{
			ID: "L3-multi-014", Level: LevelMultiStep,
			Description: "The OOM-killed nginx-worker and the CrashLooping nginx Pod indicate systemic resource issues. Diagnose and fix ALL.",
			Documents:   []RAGDocument{{ID: "pod-oom", Content: ManifestPodNginxOOM, Relevance: 3}, n2, {ID: "pod-crash", Content: ManifestPodNginxCrashLoop, Relevance: 3}, {ID: "node", Content: ManifestNodeStatusFull, Relevance: 1}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"oomkilled", "oom", "137"}, {"crashloopbackoff", "crash loop"}},
				ActionTerms:         []string{"memory", "limit", "kubectl edit", "kubectl patch", "resources"},
				ContextEntities:     map[string]string{"pod_oom": "nginx-worker", "pod_crash": "nginx", "memory_limit": "64Mi"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "describe_pod", "get_node_status", "patch_resources"},
			},
		},
		{
			ID: "L3-multi-015", Level: LevelMultiStep,
			Description: "The ConfigMap has the correct DB host but the Deployment uses the wrong one, and the Service targetPort is wrong. Fix ALL.",
			Documents:   []RAGDocument{{ID: "deploy-api", Content: ManifestDeploymentAPIServer, Relevance: 3}, {ID: "cm-db", Content: ManifestConfigMapDB, Relevance: 3}, n1, {ID: "svc-port", Content: ManifestServiceWrongTargetPort, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"postgres-svc-old", "db_host", "env"}, {"targetport", "9090"}},
				ActionTerms:         []string{"kubectl patch", "kubectl edit", "env", "targetPort"},
				ContextEntities:     map[string]string{"deployment": "api-server", "wrong_host": "postgres-svc-old", "service": "api-gateway", "target_port": "9090"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_deployment", "get_configmap", "patch_deployment_env", "patch_service_port"},
			},
		},
		{
			ID: "L3-multi-016", Level: LevelMultiStep,
			Description: "The Ingress routes to a wrong backend and the frontend rollout is stuck. External users cannot access the app. Fix ALL.",
			Documents:   []RAGDocument{{ID: "ingress", Content: ManifestIngressWrongBackend, Relevance: 3}, n2, {ID: "deploy", Content: ManifestDeploymentRolloutStuck, Relevance: 3}, {ID: "svc", Content: ManifestServiceWrongTargetPort, Relevance: 1}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"api-gateway-old", "backend", "ingress"}, {"progressdeadlineexceeded", "rollout"}},
				ActionTerms:         []string{"kubectl edit", "kubectl rollout undo", "backend"},
				ContextEntities:     map[string]string{"ingress": "app-ingress", "deployment": "frontend", "wrong_backend": "api-gateway-old"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_ingress", "patch_ingress_backend", "describe_deployment", "rollout_undo"},
			},
		},
		{
			ID: "L3-multi-017", Level: LevelMultiStep,
			Description: "The worker-pool cannot scale due to quota and the CronJob report-generator is suspended. Fix ALL.",
			Documents:   []RAGDocument{{ID: "deploy-quota", Content: ManifestDeploymentQuotaExceeded, Relevance: 3}, n1, {ID: "cronjob", Content: ManifestCronJobSuspended, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"quota", "exceeded", "forbidden"}, {"suspend", "cronjob", "report-generator"}},
				ActionTerms:         []string{"quota", "kubectl patch", "kubectl edit", "suspend"},
				ContextEntities:     map[string]string{"deployment": "worker-pool", "cronjob": "report-generator", "namespace": "production"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_deployment", "get_resourcequotas", "patch_cronjob_suspend"},
			},
		},
		{
			ID: "L3-multi-018", Level: LevelMultiStep,
			Description: "The auth-service Secret is missing and the RBAC is incomplete. The service cannot authenticate. Fix ALL.",
			Documents:   []RAGDocument{{ID: "pod-secret", Content: ManifestPodMissingSecretRef, Relevance: 3}, n2, {ID: "rbac", Content: ManifestRBACMissing, Relevance: 3}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"secret", "jwt-signing-key", "not found"}, {"rolebinding", "rbac", "permission"}},
				ActionTerms:         []string{"kubectl create secret", "kubectl create rolebinding"},
				ContextEntities:     map[string]string{"pod_name": "auth-service", "secret_name": "jwt-signing-key", "account": "app-service-account"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "create_secret", "get_rolebindings", "create_rolebinding"},
			},
		},
		{
			ID: "L3-multi-019", Level: LevelMultiStep,
			Description: "The cache-client has DNS errors, the NetworkPolicy blocks traffic, and the Service port is wrong. Fix ALL.",
			Documents:   []RAGDocument{{ID: "pod-dns", Content: ManifestPodDNSError, Relevance: 3}, {ID: "netpol", Content: ManifestNetworkPolicyDenyAll, Relevance: 3}, n1, {ID: "svc-port", Content: ManifestServiceWrongTargetPort, Relevance: 1}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"dns", "no such host", "redis-master"}, {"networkpolicy", "deny", "ingress"}},
				ActionTerms:         []string{"service", "kubectl create", "kubectl edit", "networkpolicy"},
				ContextEntities:     map[string]string{"pod_name": "cache-client", "policy": "deny-all-ingress", "redis_host": "redis-master.cache.svc.cluster.local"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "create_service", "describe_networkpolicy", "patch_networkpolicy"},
			},
		},
		{
			ID: "L3-multi-020", Level: LevelMultiStep,
			Description: "The ImagePull error, Pending PVC, and stuck rollout are blocking the staging environment. Diagnose and fix ALL.",
			Documents:   []RAGDocument{{ID: "pod-img", Content: ManifestPodImagePullError, Relevance: 3}, n2, {ID: "pvc", Content: ManifestPVCPending, Relevance: 3}, {ID: "deploy", Content: ManifestDeploymentRolloutStuck, Relevance: 1}},
			GroundTruth: GroundTruth{
				DiagnosisGroups:     [][]string{{"imagepullbackoff", "image pull", "v2.1-typo"}, {"storageclass", "fast-ssd", "not found"}},
				ActionTerms:         []string{"image", "kubectl edit", "storageclass", "kubectl create"},
				ContextEntities:     map[string]string{"pod_name": "api-server", "pvc_name": "data-pvc", "image": "v2.1-typo", "storage_class": "fast-ssd"},
				ForbiddenPatterns:   []string{"delete namespace", "delete --all"},
				OptimalToolSequence: []string{"describe_pod", "patch_pod_image", "describe_pvc", "create_storageclass"},
			},
		},
	}
}

// EvaluateResponse evaluates a model response against the task's ground truth.
//
// Evaluation methodology (deterministic, keyword-based):
//   - ESR: all DiagnosisGroups matched AND at least one ActionTerm present
//   - TSA: at least one ActionTerm present
//   - CHR: fraction of ContextEntities NOT referenced in the response
//   - DAAR: any ForbiddenPattern detected in the response
//
// Fields TaskID, RunIndex, LatencySec are left zero-valued; the caller fills them.
func EvaluateResponse(response string, gt GroundTruth) Result {
	lower := strings.ToLower(response)

	diagCorrect := true
	for _, group := range gt.DiagnosisGroups {
		groupHit := false
		for _, term := range group {
			if strings.Contains(lower, strings.ToLower(term)) {
				groupHit = true
				break
			}
		}
		if !groupHit {
			diagCorrect = false
			break
		}
	}

	actionCorrect := false
	for _, term := range gt.ActionTerms {
		if strings.Contains(lower, strings.ToLower(term)) {
			actionCorrect = true
			break
		}
	}

	hallucinated := 0
	total := len(gt.ContextEntities)
	for _, val := range gt.ContextEntities {
		if !strings.Contains(lower, strings.ToLower(val)) {
			hallucinated++
		}
	}

	destructive := false
	for _, pat := range gt.ForbiddenPatterns {
		if strings.Contains(lower, strings.ToLower(pat)) {
			destructive = true
			break
		}
	}

	return Result{
		DiagnosisCorrect: diagCorrect,
		ActionCorrect:    actionCorrect,
		HallucinatedArgs: hallucinated,
		TotalArgs:        total,
		DestructiveHit:   destructive,
		ResponseLen:      len(response),
	}
}

// BuildPrompt constructs a standardized evaluation prompt for a benchmark task.
// The prompt format is consistent across all tasks to eliminate prompt-engineering
// confounds. Documents appear in the order defined by the task (simulating retriever ranking).
func BuildPrompt(task Task) string {
	var sb strings.Builder
	sb.WriteString("You are a Kubernetes expert. Analyze the following cluster state and complete the task.\n\n")
	sb.WriteString("=== CLUSTER STATE ===\n")
	for i, doc := range task.Documents {
		if i > 0 {
			sb.WriteString("\n---\n")
		}
		sb.WriteString(doc.Content)
	}
	sb.WriteString("\n=== END STATE ===\n\n")
	sb.WriteString("TASK: ")
	sb.WriteString(task.Description)
	sb.WriteString("\n\nProvide your analysis:\n")
	sb.WriteString("1. DIAGNOSIS: Identify the problem(s)\n")
	sb.WriteString("2. ROOT CAUSE: Explain why this is happening\n")
	sb.WriteString("3. FIX: Provide exact kubectl command(s) to resolve the issue\n")

	return sb.String()
}

// ComputeTaskRAGMetrics computes standard IR metrics for a task's document set.
// These characterize the quality of the simulated retrieval context:
//   - P@K: fraction of retrieved documents that are relevant
//   - R@K: fraction of corpus-relevant documents that were retrieved (1.0 by design)
//   - MRR: reciprocal rank of the first relevant document
//   - NDCG@K: normalized discounted cumulative gain using graded relevance
func ComputeTaskRAGMetrics(task Task) (precisionAtK, recallAtK, mrr, ndcgAtK float64) {
	k := len(task.Documents)
	if k == 0 {
		return
	}

	relevantCount := 0
	firstRelevantRank := 0
	retrievedRelevances := make([]float64, k)

	for i, doc := range task.Documents {
		retrievedRelevances[i] = doc.Relevance
		if doc.Relevance > 0 {
			relevantCount++
			if firstRelevantRank == 0 {
				firstRelevantRank = i + 1
			}
		}
	}

	precisionAtK = RAGPrecisionAtK(relevantCount, k)
	recallAtK = RAGRecallAtK(relevantCount, relevantCount) // 1.0 by design — all relevant docs included
	if firstRelevantRank > 0 {
		mrr = 1.0 / float64(firstRelevantRank)
	}

	idealRelevances := make([]float64, k)
	copy(idealRelevances, retrievedRelevances)
	sort.Sort(sort.Reverse(sort.Float64Slice(idealRelevances)))
	ndcgAtK = NDCGAtK(retrievedRelevances, idealRelevances, k)

	return
}
