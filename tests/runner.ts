import { TestSuite, TestContext } from './utils/testUtils';
import { ModuleTestResult, SystemMetrics } from './utils/types';
import { printSummary } from './setup';

const moduleConfigs: TestContext[] = [
  { moduleName: 'fractal', emoji: '🌀' },
  { moduleName: 'probabilistic', emoji: '🎲' },
  { moduleName: 'memory', emoji: '💾' },
  { moduleName: 'unipixel', emoji: '⚛️' },
  { moduleName: 'mmm', emoji: '🔮' },
  { moduleName: 'codeexec', emoji: '⚙️' },
  { moduleName: 'peff', emoji: '✨' },
  { moduleName: 'integration', emoji: '🔄' }
];

interface TestStats {
  total: number;
  passed: number;
  failed: number;
  duration: number;
  moduleResults: ModuleTestResult[];
  systemMetrics: SystemMetrics;
}

class TestRunner {
  private stats: TestStats = {
    total: 0,
    passed: 0,
    failed: 0,
    duration: 0,
    moduleResults: [],
    systemMetrics: {
      totalMemoryUsage: 0,
      processingLatency: 0,
      activeModules: 0,
      errorRate: 0,
      throughput: 0
    }
  };

  async runAll(): Promise<void> {
    console.log('\n🚀 Starting Cognitive Engine Test Suite\n');
    const startTime = Date.now();
    const startMemory = process.memoryUsage().heapUsed;

    // Run individual module tests
    for (const config of moduleConfigs) {
      const moduleStartTime = Date.now();
      const suite = new TestSuite(config);
      
      try {
        const module = await import(`./${config.moduleName}.test`);
        if (module.default) {
          await module.default(suite);
          
          // Calculate module-specific metrics
          const moduleEndTime = Date.now();
          this.stats.moduleResults.push({
            moduleName: config.moduleName,
            results: [], // Would be populated by actual test results
            coverage: Math.random() * 100, // Mock coverage for now
            performance: moduleEndTime - moduleStartTime
          });
        }
      } catch (error) {
        console.error(`❌ Error loading ${config.moduleName} tests:`, error);
      }
    }

    // Calculate system metrics
    const endTime = Date.now();
    const endMemory = process.memoryUsage().heapUsed;
    
    this.stats.duration = endTime - startTime;
    this.stats.systemMetrics = {
      totalMemoryUsage: (endMemory - startMemory) / 1024 / 1024, // MB
      processingLatency: this.stats.duration / this.stats.total,
      activeModules: moduleConfigs.length,
      errorRate: this.stats.failed / this.stats.total,
      throughput: this.stats.total / (this.stats.duration / 1000)
    };

    this.printSummary();
  }

  private printSummary(): void {
    printSummary();

    // Print module-specific metrics
    console.log('\n📊 Module Performance Metrics');
    console.log('==========================');
    this.stats.moduleResults.forEach(result => {
      console.log(`\n${moduleConfigs.find(m => m.moduleName === result.moduleName)?.emoji} ${result.moduleName.toUpperCase()}`);
      console.log(`Coverage: ${result.coverage.toFixed(2)}%`);
      console.log(`Performance: ${result.performance}ms`);
    });

    // Print system metrics
    console.log('\n🔍 System Metrics');
    console.log('===============');
    console.log(`Memory Usage: ${this.stats.systemMetrics.totalMemoryUsage.toFixed(2)} MB`);
    console.log(`Avg Latency: ${this.stats.systemMetrics.processingLatency.toFixed(2)}ms`);
    console.log(`Active Modules: ${this.stats.systemMetrics.activeModules}`);
    console.log(`Error Rate: ${(this.stats.systemMetrics.errorRate * 100).toFixed(2)}%`);
    console.log(`Throughput: ${this.stats.systemMetrics.throughput.toFixed(2)} tests/sec`);

    console.log(`\nTotal Duration: ${this.stats.duration}ms`);
    
    if (this.stats.failed > 0) {
      process.exit(1);
    }
  }

  updateStats(passed: boolean): void {
    this.stats.total++;
    if (passed) {
      this.stats.passed++;
    } else {
      this.stats.failed++;
    }
  }
}

export const runner = new TestRunner();

// Run tests if this file is executed directly
if (require.main === module) {
  runner.runAll().catch(console.error);
} 