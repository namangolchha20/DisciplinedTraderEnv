import React, { useState } from 'react';
import './App.css';
import AgentTerminal from './components/AgentTerminal';
import MetricsPanel from './components/MetricsPanel';
import ChartMockup from './components/ChartMockup';
import { Activity } from 'lucide-react';

function App() {
  const [metrics, setMetrics] = useState({
    cash: 10000,
    accountValue: 10000,
    positionShares: 0,
    price: 150.25,
    regime: 'uptrend',
    pattern: 'bull_flag',
    lastAction: null
  });

  return (
    <div className="app-container" style={{ padding: '2rem', maxWidth: '1400px', margin: '0 auto', display: 'flex', flexDirection: 'column', gap: '1.5rem', minHeight: '100vh' }}>
      <header className="glass-panel" style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '1rem 2rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Activity size={32} color="var(--accent-purple)" />
          <h1 className="text-gradient" style={{ margin: 0, fontSize: '1.75rem' }}>DisciplinedTrader AI</h1>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <div style={{ width: '8px', height: '8px', borderRadius: '50%', backgroundColor: 'var(--accent-green)' }} className="animate-pulse"></div>
          <span style={{ color: 'var(--text-secondary)', fontSize: '0.9rem' }}>Environment: Ready</span>
        </div>
      </header>

      <main style={{ display: 'grid', gridTemplateColumns: '1fr 350px', gap: '1.5rem', flex: 1 }}>
        <div style={{ display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
          <ChartMockup metrics={metrics} />
          <AgentTerminal metrics={metrics} setMetrics={setMetrics} />
        </div>
        
        <aside>
          <MetricsPanel metrics={metrics} />
        </aside>
      </main>
    </div>
  );
}

export default App;
