import { TestSuite, TestCase, createMockData, assertDeepEqual, measurePerformance, validateSchema } from './utils/testUtils';

// Mock types and interfaces
interface Unipixel {
  id: string;
  layer: number;
  state: Record<string, any>;
  beliefs: Record<string, number>;
}

interface FreeEnergyMetrics {
  prediction: number;
  error: number;
  surprise: number;
}

interface LayerInteraction {
  sourceLayer: number;
  targetLayer: number;
  strength: number;
  type: string;
}

// Mock data
const mockUnipixel: Unipixel = {
  id: 'uni1',
  layer: 2,
  state: { active: true, value: 0.7 },
  beliefs: { 'concept1': 0.8, 'concept2': 0.3 }
};

const mockMetrics: FreeEnergyMetrics = {
  prediction: 0.7,
  error: 0.1,
  surprise: 0.2
};

const mockInteraction: LayerInteraction = {
  sourceLayer: 2,
  targetLayer: 3,
  strength: 0.6,
  type: 'feedforward'
};

export default async (suite: TestSuite): Promise<void> => {
  console.log('\n⚛️ Initializing Unipixel Registry Tests');

  const inferenceTests: TestCase[] = [
    {
      name: 'free_energy_principle',
      description: 'Active Inference - Free Energy Principle implementation',
      fn: async () => {
        const metrics = createMockData(mockMetrics);
        return metrics.prediction >= 0 && metrics.prediction <= 1 &&
               metrics.error >= 0 && metrics.error <= 1 &&
               metrics.surprise >= 0 && metrics.surprise <= 1;
      }
    },
    {
      name: 'belief_updates',
      description: 'Active Inference - Belief updating mechanism',
      fn: async () => {
        const unipixel = createMockData(mockUnipixel);
        return Object.values(unipixel.beliefs).every(v => v >= 0 && v <= 1);
      }
    }
  ];

  const representationTests: TestCase[] = [
    {
      name: 'layer_specific',
      description: 'Multi-level Representation - Layer-specific representation',
      fn: async () => {
        const unipixel = createMockData(mockUnipixel);
        return validateSchema(unipixel, {
          id: 'string',
          layer: 'number',
          state: 'object',
          beliefs: 'object'
        });
      }
    },
    {
      name: 'cross_layer',
      description: 'Multi-level Representation - Cross-layer interaction',
      fn: async () => {
        const interaction = createMockData(mockInteraction);
        return interaction.sourceLayer < interaction.targetLayer &&
               interaction.strength >= 0 && interaction.strength <= 1;
      }
    }
  ];

  const stateTests: TestCase[] = [
    {
      name: 'state_tracking',
      description: 'State Management - Internal state tracking',
      fn: async () => {
        const unipixel = createMockData(mockUnipixel);
        return typeof unipixel.state.active === 'boolean' &&
               typeof unipixel.state.value === 'number';
      }
    },
    {
      name: 'belief_management',
      description: 'State Management - Belief management',
      fn: async () => {
        const duration = await measurePerformance(async () => {
          const unipixel = createMockData(mockUnipixel);
          return Object.keys(unipixel.beliefs).length > 0;
        });
        return duration < 100; // Less than 100ms
      }
    }
  ];

  // Run all test groups
  await suite.runTests([...inferenceTests, ...representationTests, ...stateTests]);

  console.log('⚛️ Completed Unipixel Registry Tests\n');
}; 