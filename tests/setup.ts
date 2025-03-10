export interface TestReport {
  moduleName: string;
  passed: boolean;
  details: string;
}

export const testReports: TestReport[] = [];

const moduleEmojis = {
  fractal: '🌀',
  probabilistic: '🎲',
  memory: '💾',
  unipixel: '⚛️',
  mmm: '🔮',
  codeexec: '⚙️',
  peff: '✨'
};

export const reportTest = (moduleName: string, passed: boolean, details: string) => {
  const emoji = moduleEmojis[moduleName as keyof typeof moduleEmojis] || '📋';
  const status = passed ? '✅' : '❌';
  
  console.log(`\n${emoji} Testing ${moduleName.toUpperCase()} Module ${status}`);
  console.log(`${details}\n`);
  
  testReports.push({ moduleName, passed, details });
};

export const printSummary = () => {
  console.log('\n📊 Test Summary Report 📊');
  console.log('========================');
  
  const totalTests = testReports.length;
  const passedTests = testReports.filter(r => r.passed).length;
  
  testReports.forEach(({ moduleName, passed, details }) => {
    const emoji = moduleEmojis[moduleName as keyof typeof moduleEmojis] || '📋';
    console.log(`${emoji} ${moduleName}: ${passed ? '✅' : '❌'}`);
  });
  
  console.log('\n📈 Statistics:');
  console.log(`Total Tests: ${totalTests}`);
  console.log(`Passed: ${passedTests}`);
  console.log(`Failed: ${totalTests - passedTests}`);
  console.log(`Success Rate: ${((passedTests / totalTests) * 100).toFixed(2)}%`);
}; 