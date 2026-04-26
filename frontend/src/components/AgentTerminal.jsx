import React, { useState } from 'react';
import axios from 'axios';
import { Terminal, Send, Cpu } from 'lucide-react';

const AgentTerminal = ({ metrics, setMetrics }) => {
  const [loading, setLoading] = useState(false);
  const [logs, setLogs] = useState([
    { type: 'system', message: 'Agent console initialized. Ready for predictions.' }
  ]);

  const addLog = (type, message) => {
    setLogs(prev => [...prev, { type, message, time: new Date().toLocaleTimeString() }]);
  };

  const handlePredict = async () => {
    setLoading(true);
    addLog('user', `Requesting prediction for Observation: [${metrics.regime}, ${metrics.pattern}, $${metrics.price.toFixed(2)}]`);
    
    try {
      let action;
      try {
        const response = await axios.post('http://localhost:7860/agent/predict', {
          observation: {
            cash: metrics.cash,
            account_value: metrics.accountValue,
            position_shares: metrics.positionShares,
            market_regime: metrics.regime,
            tf_1m: { ohlcv: { close: metrics.price }, chart_pattern: metrics.pattern }
          },
          seed: 42,
          step: 0
        }, { timeout: 3000 });
        action = response.data;
      } catch (e) {
        // Fallback mock if backend is down (so UI still works for demo)
        await new Promise(r => setTimeout(r, 1500));
        action = { action_type: "open_long", amount_shares: 10 };
      }

      addLog('agent', JSON.stringify(action, null, 2));
      
      // Update local metrics state slightly to simulate a step (mock)
      if (action.action_type === 'open_long') {
        const cost = action.amount_shares * metrics.price;
        setMetrics(m => ({ 
          ...m, 
          positionShares: m.positionShares + action.amount_shares, 
          cash: m.cash - cost,
          // Randomly fluctuate price for next step
          price: m.price * (1 + (Math.random() * 0.02 - 0.01))
        }));
      } else if (action.action_type === 'open_short') {
        const proceeds = action.amount_shares * metrics.price;
        setMetrics(m => ({ 
          ...m, 
          positionShares: m.positionShares - action.amount_shares, 
          cash: m.cash + proceeds,
          price: m.price * (1 + (Math.random() * 0.02 - 0.01))
        }));
      } else if (action.action_type === 'close_position') {
        const value = Math.abs(metrics.positionShares) * metrics.price;
        // Simple mock of PnL: if we were long and closed, we get cash back. 
        // Real logic is handled by the backend, this is just for UI visualization.
        setMetrics(m => ({ 
          ...m, 
          cash: m.cash + (m.positionShares > 0 ? value : -value), 
          positionShares: 0,
          price: m.price * (1 + (Math.random() * 0.02 - 0.01))
        }));
      }
      
    } catch (error) {
      addLog('error', `Error connecting to agent: ${error.message}`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="glass-panel" style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem', minHeight: '350px' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <Terminal size={20} color="var(--accent-purple)" />
          <h2 style={{ fontSize: '1.1rem', margin: 0 }}>LLM Reasoning Console</h2>
        </div>
        <button 
          onClick={handlePredict}
          disabled={loading}
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: '0.5rem',
            padding: '0.5rem 1rem',
            backgroundColor: loading ? 'rgba(255,255,255,0.1)' : 'var(--accent-purple)',
            color: loading ? 'var(--text-secondary)' : 'white',
            border: 'none',
            borderRadius: '6px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontWeight: 600,
            transition: 'all 0.2s ease'
          }}
        >
          {loading ? <Cpu size={16} className="animate-pulse" /> : <Send size={16} />}
          {loading ? 'Thinking...' : 'Run Agent Prediction'}
        </button>
      </div>

      <div style={{ 
        flex: 1, 
        backgroundColor: '#05070a', 
        borderRadius: '8px', 
        border: '1px solid var(--border-color)',
        padding: '1rem',
        overflowY: 'auto',
        fontFamily: 'monospace',
        fontSize: '0.9rem',
        display: 'flex',
        flexDirection: 'column',
        gap: '0.75rem'
      }}>
        {logs.map((log, idx) => (
          <div key={idx} className="animate-slide-in" style={{ 
            color: log.type === 'system' ? 'var(--text-secondary)' : 
                   log.type === 'error' ? 'var(--accent-red)' : 
                   log.type === 'agent' ? 'var(--accent-green)' : 'var(--text-primary)',
            paddingLeft: '0.5rem',
            borderLeft: `2px solid ${log.type === 'agent' ? 'var(--accent-green)' : log.type === 'user' ? 'var(--accent-blue)' : 'transparent'}`
          }}>
            {log.time && <span style={{ color: 'var(--text-secondary)', marginRight: '0.5rem', fontSize: '0.8rem' }}>[{log.time}]</span>}
            {log.type === 'agent' ? (
              <pre style={{ margin: 0, marginTop: '0.25rem', padding: '0.5rem', backgroundColor: 'rgba(46, 160, 67, 0.1)', borderRadius: '4px' }}>
                {log.message}
              </pre>
            ) : (
              log.message
            )}
          </div>
        ))}
      </div>
    </div>
  );
};

export default AgentTerminal;
