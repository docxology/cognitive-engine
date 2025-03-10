// Common test result types
export interface TestResult {
  passed: boolean;
  duration: number;
  error?: string;
  details?: string;
  metadata?: Record<string, any>;
}

export interface ModuleTestResult {
  moduleName: string;
  results: TestResult[];
  coverage: number;
  performance: number;
}

export interface CrossModuleTestResult {
  sourceModule: string;
  targetModule: string;
  testName: string;
  passed: boolean;
  details: string;
}

// Module-specific interfaces
export interface ModuleState {
  isInitialized: boolean;
  isHealthy: boolean;
  lastUpdate: number;
  metrics: Record<string, number>;
  errorState?: {
    hasError: boolean;
    errorCount: number;
    lastError?: string;
  };
}

export interface ModuleInteraction {
  source: string;
  target: string;
  type: string;
  data: any;
  timestamp: number;
}

// System-wide metrics
export interface SystemMetrics {
  totalMemoryUsage: number;
  processingLatency: number;
  activeModules: number;
  errorRate: number;
  throughput: number;
  recoveryEvents?: number;
}

// Integration test configuration
export interface IntegrationConfig {
  modules: string[];
  dependencies: Record<string, string[]>;
  requiredInteractions: ModuleInteraction[];
}

// Performance benchmarks
export interface PerformanceBenchmark {
  operation: string;
  expectedLatency: number;
  maxMemory: number;
  minThroughput: number;
}

// Neural Network types
export interface LayerConfig {
  size: number;
  type: 'input' | 'hidden' | 'output';
  activation?: string;
  dropout?: number;
}

export interface NeuralNetworkConfig {
  layers: LayerConfig[];
  learningRate: number;
  batchSize: number;
  optimizer: string;
  activationFunction: string;
  fallbackMode?: boolean;
  maxRetries?: number;
  recoveryPath?: string;
}

export interface TrainingData {
  input: string;
  expectedOutput: string;
  metadata: {
    timestamp: number;
    context: string;
    confidence: number;
  };
}

export interface ModelMetrics {
  loss: number;
  accuracy: number;
  epochNumber: number;
  learningCurve: number[];
  confusionMatrix: Map<string, number>;
  memoryUsage: number;
  errorCount?: number;
  recoveryEvents?: number;
  lastError?: string;
  fallbackMode?: boolean;
}

export interface ErrorRecord {
  timestamp: number;
  error: string;
  errorType: string;
  recoveryAction?: string;
  recoverySuccess?: boolean;
} 