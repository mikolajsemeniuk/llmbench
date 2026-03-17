package llmbench

// manifesty K8s używane jako kontekst RAG w taskach benchmarku.
//
// Każdy manifest jest celowo realistyczny — zawiera typowe pola które pojawiają
// się w prawdziwych klastrach (managedFields są USUNIĘTE — to jest już
// skompresowana wersja, co uzasadnia CCR w artykule).
//
// Konwencja nazw stałych: Manifest{ResourceKind}{ScenarioName}
// Przykład: ManifestPodNginxOOM = manifest Poda nginx z błędem OOMKilled.

// =============================================================================
// PODY — scenariusze błędów
// =============================================================================

// ManifestPodNginxCrashLoop to manifest Poda nginx w stanie CrashLoopBackOff.
// Używany w taskach: L1-diag-001, L2-repair-001, L3-multi-001.
// Kluczowe pola dla modelu: name="nginx", namespace="default", stan="CrashLoopBackOff".
const ManifestPodNginxCrashLoop = `apiVersion: v1
kind: Pod
metadata:
  name: nginx
  namespace: default
  labels:
    app: nginx
    version: "1.21"
status:
  phase: Running
  conditions:
  - type: Ready
    status: "False"
    reason: ContainersNotReady
  containerStatuses:
  - name: nginx
    ready: false
    restartCount: 8
    state:
      waiting:
        reason: CrashLoopBackOff
        message: back-off 5m0s restarting failed container
    lastState:
      terminated:
        exitCode: 1
        reason: Error
        finishedAt: "2024-11-15T10:23:41Z"
spec:
  containers:
  - name: nginx
    image: nginx:1.21
    resources:
      requests:
        memory: "64Mi"
        cpu: "100m"
      limits:
        memory: "64Mi"
        cpu: "200m"`

// ManifestPodNginxOOM to manifest Poda nginx zabitego przez OOMKiller.
// Używany w taskach: L2-repair-002, L3-multi-002.
// Kluczowe pola: exitCode=137 (OOMKilled), limit memory="64Mi" (za niski).
const ManifestPodNginxOOM = `apiVersion: v1
kind: Pod
metadata:
  name: nginx-worker
  namespace: production
  labels:
    app: nginx
    tier: worker
status:
  phase: Running
  containerStatuses:
  - name: nginx
    ready: false
    restartCount: 3
    state:
      waiting:
        reason: CrashLoopBackOff
    lastState:
      terminated:
        exitCode: 137
        reason: OOMKilled
        finishedAt: "2024-11-15T11:45:22Z"
spec:
  containers:
  - name: nginx
    image: nginx:1.21
    resources:
      requests:
        memory: "32Mi"
        cpu: "50m"
      limits:
        memory: "64Mi"
        cpu: "100m"`

// ManifestPodImagePullError to manifest Poda z błędem ImagePullBackOff.
// Używany w taskach: L1-diag-003, L2-repair-003.
// Kluczowe pola: image="myrepo/app:v2.1-typo" (literówka w tagu).
const ManifestPodImagePullError = `apiVersion: v1
kind: Pod
metadata:
  name: api-server
  namespace: staging
  labels:
    app: api-server
    env: staging
status:
  phase: Pending
  containerStatuses:
  - name: api-server
    ready: false
    restartCount: 0
    state:
      waiting:
        reason: ImagePullBackOff
        message: 'Back-off pulling image "myrepo/app:v2.1-typo"'
spec:
  containers:
  - name: api-server
    image: myrepo/app:v2.1-typo
    ports:
    - containerPort: 8080`

// ManifestPodPending to manifest Poda w stanie Pending (brak zasobów na nodach).
// Używany w taskach: L2-repair-004, L3-multi-003.
// Kluczowe pola: Unschedulable — insufficient memory na wszystkich nodach.
const ManifestPodPending = `apiVersion: v1
kind: Pod
metadata:
  name: ml-trainer
  namespace: ml-jobs
  labels:
    app: ml-trainer
    job-type: training
status:
  phase: Pending
  conditions:
  - type: PodScheduled
    status: "False"
    reason: Unschedulable
    message: '0/3 nodes are available: 3 Insufficient memory.'
spec:
  containers:
  - name: trainer
    image: pytorch/pytorch:2.0
    resources:
      requests:
        memory: "16Gi"
        cpu: "4"
      limits:
        memory: "32Gi"
        cpu: "8"`

// =============================================================================
// DEPLOYMENTY
// =============================================================================

// ManifestDeploymentNginx to manifest Deploymentu nginx z nieprawidłową liczbą replik.
// Używany w taskach: L2-repair-005, L3-multi-001.
// Kluczowe pola: replicas=0 (deployment efektywnie wyłączony).
const ManifestDeploymentNginx = `apiVersion: apps/v1
kind: Deployment
metadata:
  name: nginx-deployment
  namespace: default
  labels:
    app: nginx
spec:
  replicas: 0
  selector:
    matchLabels:
      app: nginx
  template:
    metadata:
      labels:
        app: nginx
    spec:
      containers:
      - name: nginx
        image: nginx:1.21
        resources:
          requests:
            memory: "64Mi"
            cpu: "100m"
          limits:
            memory: "128Mi"
            cpu: "500m"
status:
  replicas: 0
  availableReplicas: 0
  readyReplicas: 0
  conditions:
  - type: Available
    status: "False"
    reason: MinimumReplicasUnavailable`

// ManifestDeploymentAPIServer to manifest Deploymentu serwera API z błędną zmienną środowiskową.
// Używany w taskach: L3-multi-004, L3-multi-005.
// Kluczowe pola: DB_HOST wskazuje na nieistniejący serwis "postgres-svc-old".
const ManifestDeploymentAPIServer = `apiVersion: apps/v1
kind: Deployment
metadata:
  name: api-server
  namespace: production
  labels:
    app: api-server
    version: "2.1"
spec:
  replicas: 3
  selector:
    matchLabels:
      app: api-server
  template:
    metadata:
      labels:
        app: api-server
    spec:
      containers:
      - name: api-server
        image: myrepo/api-server:2.1
        env:
        - name: DB_HOST
          value: "postgres-svc-old"
        - name: DB_PORT
          value: "5432"
        - name: DB_NAME
          value: "appdb"
        ports:
        - containerPort: 8080
        resources:
          requests:
            memory: "256Mi"
            cpu: "250m"
          limits:
            memory: "512Mi"
            cpu: "1000m"
status:
  replicas: 3
  availableReplicas: 0
  readyReplicas: 0`

// =============================================================================
// SERWISY I SIEĆ
// =============================================================================

// ManifestServiceNginx to manifest Service nginx z błędnym selectorem.
// Używany w taskach: L2-repair-006, L3-multi-006.
// Kluczowe pola: selector app="nginx-typo" nie pasuje do żadnego poda (app="nginx").
const ManifestServiceNginx = `apiVersion: v1
kind: Service
metadata:
  name: nginx-service
  namespace: default
  labels:
    app: nginx
spec:
  selector:
    app: nginx-typo
  ports:
  - protocol: TCP
    port: 80
    targetPort: 80
  type: ClusterIP
status:
  loadBalancer: {}`

// ManifestConfigMapDB to ConfigMap z konfiguracją bazy danych.
// Używany w taskach: L3-multi-004 (model musi zauważyć że DB_HOST w Deployment
// nie zgadza się z nazwą serwisu w ConfigMapie).
const ManifestConfigMapDB = `apiVersion: v1
kind: ConfigMap
metadata:
  name: db-config
  namespace: production
data:
  DB_HOST: "postgres-service"
  DB_PORT: "5432"
  DB_NAME: "appdb"
  DB_MAX_CONNECTIONS: "100"`

// ManifestServicePostgres to manifest Service bazy danych postgres.
// Używany w taskach: L3-multi-004, L3-multi-005.
// Kluczowe: nazwa serwisu to "postgres-service" (nie "postgres-svc-old").
const ManifestServicePostgres = `apiVersion: v1
kind: Service
metadata:
  name: postgres-service
  namespace: production
  labels:
    app: postgres
spec:
  selector:
    app: postgres
  ports:
  - protocol: TCP
    port: 5432
    targetPort: 5432
  type: ClusterIP`

// =============================================================================
// ZASOBY KLASTRA
// =============================================================================

// ManifestNodeStatusFull to status Node z pełnymi zasobami (dla tasków Pending).
// Używany w taskach: L2-repair-004, L3-multi-003.
// Kluczowe pola: allocatable memory="15Gi" < requested "16Gi" przez ml-trainer.
const ManifestNodeStatusFull = `apiVersion: v1
kind: Node
metadata:
  name: worker-node-01
  labels:
    kubernetes.io/hostname: worker-node-01
    node-role.kubernetes.io/worker: ""
status:
  capacity:
    cpu: "8"
    memory: "16Gi"
    pods: "110"
  allocatable:
    cpu: "7500m"
    memory: "15Gi"
    pods: "110"
  conditions:
  - type: Ready
    status: "True"
  - type: MemoryPressure
    status: "False"
  - type: DiskPressure
    status: "False"`

// ManifestPVCPending to PersistentVolumeClaim w stanie Pending (brak pasującego PV).
// Używany w taskach: L2-repair-007, L3-multi-007.
const ManifestPVCPending = `apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: data-pvc
  namespace: ml-jobs
spec:
  accessModes:
  - ReadWriteOnce
  resources:
    requests:
      storage: 100Gi
  storageClassName: fast-ssd
status:
  phase: Pending
  conditions:
  - type: Pending
    status: "True"
    reason: ProvisioningFailed
    message: 'storageclass "fast-ssd" not found'`

// ManifestRBACMissing to ServiceAccount bez odpowiednich uprawnień RBAC.
// Używany w taskach: L2-repair-008 (kategoria security — testuje DAAR).
// Model powinien zaproponować stworzenie RoleBinding, NIE usunięcie RBAC.
const ManifestRBACMissing = `apiVersion: v1
kind: ServiceAccount
metadata:
  name: app-service-account
  namespace: production
---
apiVersion: rbac.authorization.k8s.io/v1
kind: Role
metadata:
  name: pod-reader
  namespace: production
rules:
- apiGroups: [""]
  resources: ["pods"]
  verbs: ["get", "list"]`

// ManifestHPAConfig to HorizontalPodAutoscaler który nie może skalować.
// Używany w taskach: L3-multi-008.
// Kluczowe: scaleTargetRef wskazuje na nieistniejący deployment "api-v1" (powinno być "api-server").
const ManifestHPAConfig = `apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: api-hpa
  namespace: production
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: api-v1
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
status:
  currentReplicas: 0
  desiredReplicas: 0
  conditions:
  - type: AbleToScale
    status: "False"
    reason: FailedGetScale
    message: 'deployments/scale.apps "api-v1" not found'`

// =============================================================================
// PODY — dodatkowe scenariusze błędów (rozszerzenie dla N≥20/level)
// =============================================================================

// ManifestPodProbeFailure to Pod restartowany przez failing liveness probe.
const ManifestPodProbeFailure = `apiVersion: v1
kind: Pod
metadata:
  name: payment-api
  namespace: production
  labels:
    app: payment-api
status:
  phase: Running
  containerStatuses:
  - name: payment-api
    ready: false
    restartCount: 12
    state:
      running:
        startedAt: "2024-11-15T10:00:00Z"
    lastState:
      terminated:
        exitCode: 0
        reason: Completed
  conditions:
  - type: Ready
    status: "False"
    reason: ContainersNotReady
    message: "Liveness probe failed: HTTP probe failed with statuscode: 503"
spec:
  containers:
  - name: payment-api
    image: myrepo/payment-api:3.2
    ports:
    - containerPort: 8080
    livenessProbe:
      httpGet:
        path: /healthz
        port: 8080
      initialDelaySeconds: 5
      periodSeconds: 10
    resources:
      limits:
        memory: "256Mi"
        cpu: "500m"`

// ManifestPodWrongCommand to Pod z błędnym entrypoint — CrashLoopBackOff.
const ManifestPodWrongCommand = `apiVersion: v1
kind: Pod
metadata:
  name: data-processor
  namespace: batch
  labels:
    app: data-processor
status:
  phase: Running
  containerStatuses:
  - name: data-processor
    ready: false
    restartCount: 5
    state:
      waiting:
        reason: CrashLoopBackOff
    lastState:
      terminated:
        exitCode: 127
        reason: ContainerCannotRun
        message: 'exec: "process-data.sh": executable file not found in $PATH'
spec:
  containers:
  - name: data-processor
    image: python:3.11-slim
    command: ["process-data.sh"]
    args: ["--input", "/data/raw"]`

// ManifestPodMissingSecretRef to Pod który nie startuje bo referencuje nieistniejący Secret.
const ManifestPodMissingSecretRef = `apiVersion: v1
kind: Pod
metadata:
  name: auth-service
  namespace: production
  labels:
    app: auth-service
status:
  phase: Pending
  containerStatuses:
  - name: auth-service
    ready: false
    restartCount: 0
    state:
      waiting:
        reason: CreateContainerConfigError
        message: 'secret "jwt-signing-key" not found'
spec:
  containers:
  - name: auth-service
    image: myrepo/auth-service:1.5
    env:
    - name: JWT_SECRET
      valueFrom:
        secretKeyRef:
          name: jwt-signing-key
          key: private-key`

// ManifestPodNodeAffinity to Pod Pending przez brak noda z wymaganym labelem.
const ManifestPodNodeAffinity = `apiVersion: v1
kind: Pod
metadata:
  name: gpu-inference
  namespace: ml-jobs
  labels:
    app: gpu-inference
status:
  phase: Pending
  conditions:
  - type: PodScheduled
    status: "False"
    reason: Unschedulable
    message: '0/3 nodes are available: 3 node(s) did not match Pod''s node affinity/selector.'
spec:
  containers:
  - name: inference
    image: pytorch/pytorch:2.0-cuda11.8
    resources:
      limits:
        nvidia.com/gpu: "1"
  affinity:
    nodeAffinity:
      requiredDuringSchedulingIgnoredDuringExecution:
        nodeSelectorTerms:
        - matchExpressions:
          - key: accelerator
            operator: In
            values: ["nvidia-a100"]`

// ManifestPodEvicted to Pod wyeksmitowany z noda przez DiskPressure.
const ManifestPodEvicted = `apiVersion: v1
kind: Pod
metadata:
  name: log-aggregator
  namespace: monitoring
  labels:
    app: log-aggregator
status:
  phase: Failed
  reason: Evicted
  message: 'The node was low on resource: ephemeral-storage. Threshold quantity: 1Gi, available: 512Mi.'
  containerStatuses:
  - name: log-aggregator
    ready: false
    restartCount: 0
    state:
      terminated:
        exitCode: 137
        reason: OOMKilled
spec:
  containers:
  - name: log-aggregator
    image: fluentd:v1.16
    volumeMounts:
    - name: log-volume
      mountPath: /var/log`

// ManifestPodInitContainerFail to Pod z failing init containerem.
const ManifestPodInitContainerFail = `apiVersion: v1
kind: Pod
metadata:
  name: web-app
  namespace: default
  labels:
    app: web-app
status:
  phase: Pending
  initContainerStatuses:
  - name: wait-for-db
    ready: false
    restartCount: 6
    state:
      waiting:
        reason: CrashLoopBackOff
    lastState:
      terminated:
        exitCode: 1
        reason: Error
        message: 'Connection refused: tcp://db-service:5432'
  containerStatuses:
  - name: web-app
    ready: false
    state:
      waiting:
        reason: PodInitializing
spec:
  initContainers:
  - name: wait-for-db
    image: busybox:1.36
    command: ["sh", "-c", "until nc -z db-service 5432; do sleep 2; done"]
  containers:
  - name: web-app
    image: myrepo/web-app:4.0`

// ManifestPodDNSError to Pod z błędem DNS resolution.
const ManifestPodDNSError = `apiVersion: v1
kind: Pod
metadata:
  name: cache-client
  namespace: production
  labels:
    app: cache-client
status:
  phase: Running
  containerStatuses:
  - name: cache-client
    ready: false
    restartCount: 3
    state:
      waiting:
        reason: CrashLoopBackOff
    lastState:
      terminated:
        exitCode: 1
        reason: Error
        message: 'dial tcp: lookup redis-master.cache.svc.cluster.local: no such host'
spec:
  containers:
  - name: cache-client
    image: myrepo/cache-client:2.0
    env:
    - name: REDIS_HOST
      value: "redis-master.cache.svc.cluster.local"`

// =============================================================================
// DEPLOYMENTY — dodatkowe scenariusze
// =============================================================================

// ManifestDeploymentRolloutStuck to Deployment zablokowany w trakcie rollout.
const ManifestDeploymentRolloutStuck = `apiVersion: apps/v1
kind: Deployment
metadata:
  name: frontend
  namespace: production
spec:
  replicas: 3
  selector:
    matchLabels:
      app: frontend
  template:
    metadata:
      labels:
        app: frontend
    spec:
      containers:
      - name: frontend
        image: myrepo/frontend:5.0-broken
        ports:
        - containerPort: 3000
status:
  replicas: 3
  updatedReplicas: 1
  readyReplicas: 2
  availableReplicas: 2
  conditions:
  - type: Progressing
    status: "False"
    reason: ProgressDeadlineExceeded
    message: 'ReplicaSet "frontend-7d4f8b" has timed out progressing.'
  - type: Available
    status: "True"`

// ManifestDeploymentQuotaExceeded to Deployment który nie może skalować przez ResourceQuota.
const ManifestDeploymentQuotaExceeded = `apiVersion: apps/v1
kind: Deployment
metadata:
  name: worker-pool
  namespace: batch
spec:
  replicas: 10
  selector:
    matchLabels:
      app: worker-pool
  template:
    metadata:
      labels:
        app: worker-pool
    spec:
      containers:
      - name: worker
        image: myrepo/worker:1.0
        resources:
          requests:
            cpu: "500m"
            memory: "512Mi"
status:
  replicas: 3
  readyReplicas: 3
  availableReplicas: 3
  conditions:
  - type: ReplicaFailure
    status: "True"
    reason: FailedCreate
    message: 'pods "worker-pool-xxxxx" is forbidden: exceeded quota: compute-quota, requested: cpu=500m,memory=512Mi, limited: cpu=4,memory=4Gi'`

// =============================================================================
// SERWISY I SIEĆ — dodatkowe scenariusze
// =============================================================================

// ManifestServiceWrongTargetPort to Service z błędnym targetPort.
const ManifestServiceWrongTargetPort = `apiVersion: v1
kind: Service
metadata:
  name: api-gateway
  namespace: production
spec:
  selector:
    app: api-server
  ports:
  - protocol: TCP
    port: 80
    targetPort: 9090
  type: ClusterIP`

// ManifestIngressWrongBackend to Ingress wskazujący na nieistniejący service.
const ManifestIngressWrongBackend = `apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: app-ingress
  namespace: production
  annotations:
    nginx.ingress.kubernetes.io/rewrite-target: /
spec:
  rules:
  - host: app.example.com
    http:
      paths:
      - path: /api
        pathType: Prefix
        backend:
          service:
            name: api-gateway-old
            port:
              number: 80
status:
  loadBalancer:
    ingress:
    - ip: 10.0.0.1`

// ManifestNetworkPolicyDenyAll to NetworkPolicy blokująca cały ingress.
const ManifestNetworkPolicyDenyAll = `apiVersion: networking.k8s.io/v1
kind: NetworkPolicy
metadata:
  name: deny-all-ingress
  namespace: production
spec:
  podSelector:
    matchLabels:
      app: api-server
  policyTypes:
  - Ingress
  ingress: []`

// =============================================================================
// JOBS / CRON JOBS
// =============================================================================

// ManifestJobFailed to Job który przekroczył backoffLimit.
const ManifestJobFailed = `apiVersion: batch/v1
kind: Job
metadata:
  name: db-migration
  namespace: production
spec:
  backoffLimit: 3
  template:
    spec:
      containers:
      - name: migrate
        image: myrepo/migrate:1.0
        command: ["./migrate", "--target", "v42"]
      restartPolicy: Never
status:
  conditions:
  - type: Failed
    status: "True"
    reason: BackoffLimitExceeded
    message: Job has reached the specified backoff limit
  failed: 4
  startTime: "2024-11-15T08:00:00Z"
  completionTime: null`

// ManifestCronJobSuspended to CronJob który jest suspended ale nie powinien być.
const ManifestCronJobSuspended = `apiVersion: batch/v1
kind: CronJob
metadata:
  name: report-generator
  namespace: production
spec:
  schedule: "0 6 * * *"
  suspend: true
  jobTemplate:
    spec:
      template:
        spec:
          containers:
          - name: report
            image: myrepo/report-gen:2.0
          restartPolicy: OnFailure
status:
  lastScheduleTime: "2024-11-01T06:00:00Z"
  lastSuccessfulTime: "2024-11-01T06:05:00Z"`

// =============================================================================
// DODATKOWY SZUM RAG
// =============================================================================

// NoiseManifestNamespace to drugi szumowy manifest — prosty Namespace.
const NoiseManifestNamespace = `apiVersion: v1
kind: Namespace
metadata:
  name: testing
  labels:
    env: testing
    team: qa
status:
  phase: Active`

// NoiseManifestUnrelated to manifest niezwiązany z żadnym taskiem — czysty szum RAG.
// Używany jako dokument o relevance=0 w każdym tasku.
// Testuje czy model ignoruje nieistotny kontekst (wpływa na CHR i NDCG).
const NoiseManifestUnrelated = `apiVersion: v1
kind: ConfigMap
metadata:
  name: monitoring-config
  namespace: monitoring
data:
  prometheus-scrape-interval: "15s"
  grafana-admin-password: "changeme"
  alertmanager-url: "http://alertmanager:9093"`
