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
