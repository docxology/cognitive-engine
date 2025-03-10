import { reportTest } from '../setup';

export interface TestContext {
  moduleName: string;
  emoji: string;
}

export interface TestCase {
  name: string;
  fn: () => Promise<boolean> | boolean;
  description: string;
}

export class TestSuite {
  private context: TestContext;

  constructor(context: TestContext) {
    this.context = context;
  }

  async runTest(testCase: TestCase): Promise<void> {
    try {
      const result = await testCase.fn();
      reportTest(this.context.moduleName, result, testCase.description);
    } catch (error) {
      const errorMessage = error instanceof Error ? error.message : String(error);
      console.error(`${this.context.emoji} Error in test ${testCase.name}:`, error);
      reportTest(this.context.moduleName, false, `${testCase.description} (Error: ${errorMessage})`);
    }
  }

  async runTests(testCases: TestCase[]): Promise<void> {
    for (const testCase of testCases) {
      await this.runTest(testCase);
    }
  }
}

export const createMockData = <T>(template: T): T => {
  return { ...template };
};

export const assertDeepEqual = (actual: any, expected: any): boolean => {
  try {
    expect(actual).toEqual(expected);
    return true;
  } catch {
    return false;
  }
};

export const measurePerformance = async (fn: () => Promise<any>): Promise<number> => {
  const start = process.hrtime.bigint();
  await fn();
  const end = process.hrtime.bigint();
  return Number(end - start) / 1_000_000; // Convert to milliseconds
};

export const validateSchema = (data: any, schema: any): boolean => {
  try {
    for (const [key, type] of Object.entries(schema)) {
      if (typeof data[key] !== type) {
        return false;
      }
    }
    return true;
  } catch {
    return false;
  }
}; 