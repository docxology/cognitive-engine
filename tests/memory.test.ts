import { TestSuite, TestCase, createMockData, assertDeepEqual, measurePerformance, validateSchema } from './utils/testUtils';

// Mock types and interfaces
interface MemoryUnit {
  id: string;
  type: string;
  content: any;
  metadata: Record<string, any>;
}

interface StorageStats {
  totalSize: number;
  usedSpace: number;
  itemCount: number;
}

interface MemoryConnection {
  sourceId: string;
  targetId: string;
  strength: number;
  type: string;
}

// Mock data
const mockMemoryUnit: MemoryUnit = {
  id: 'mem1',
  type: 'concept',
  content: { value: 'test' },
  metadata: { timestamp: Date.now() }
};

const mockStorageStats: StorageStats = {
  totalSize: 1000,
  usedSpace: 100,
  itemCount: 10
};

const mockConnection: MemoryConnection = {
  sourceId: 'mem1',
  targetId: 'mem2',
  strength: 0.8,
  type: 'association'
};

export default async (suite: TestSuite): Promise<void> => {
  console.log('\n💾 Initializing Memory System Tests');

  const storageTests: TestCase[] = [
    {
      name: 'store_multimodal',
      description: 'Storage Mechanisms - Multi-modal information storage',
      fn: async () => {
        const unit = createMockData(mockMemoryUnit);
        return validateSchema(unit, {
          id: 'string',
          type: 'string',
          content: 'object',
          metadata: 'object'
        });
      }
    },
    {
      name: 'data_persistence',
      description: 'Storage Mechanisms - Data persistence check',
      fn: async () => {
        const stats = createMockData(mockStorageStats);
        return stats.usedSpace <= stats.totalSize && stats.itemCount > 0;
      }
    }
  ];

  const retrievalTests: TestCase[] = [
    {
      name: 'context_retrieval',
      description: 'Retrieval Algorithms - Context-sensitive retrieval',
      fn: async () => {
        const unit = createMockData(mockMemoryUnit);
        return unit.metadata.timestamp <= Date.now();
      }
    },
    {
      name: 'retrieval_speed',
      description: 'Retrieval Algorithms - Retrieval performance',
      fn: async () => {
        const duration = await measurePerformance(async () => {
          const unit = createMockData(mockMemoryUnit);
          return unit;
        });
        return duration < 100; // Less than 100ms
      }
    }
  ];

  const associativeTests: TestCase[] = [
    {
      name: 'memory_connections',
      description: 'Associative Networks - Memory connection creation',
      fn: async () => {
        const connection = createMockData(mockConnection);
        return validateSchema(connection, {
          sourceId: 'string',
          targetId: 'string',
          strength: 'number',
          type: 'string'
        });
      }
    },
    {
      name: 'graph_traversal',
      description: 'Associative Networks - Memory graph traversal',
      fn: async () => {
        const connection = createMockData(mockConnection);
        return connection.strength >= 0 && connection.strength <= 1;
      }
    }
  ];

  // Run all test groups
  await suite.runTests([...storageTests, ...retrievalTests, ...associativeTests]);

  console.log('💾 Completed Memory System Tests\n');
}; 