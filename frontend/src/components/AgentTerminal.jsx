import React, { useState } from 'react';
import axios from 'axios';
import { Terminal, Send, Cpu } from 'lucide-react';

const AgentTerminal = ({ sessionId, metrics, setMetrics }) => {
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
          observation: metrics.rawObservation,
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
      
      // Advance the real environment!
      if (sessionId) {
        try {
          const stepRes = await axios.post(`http://localhost:7860/api/step/${sessionId}`, action);
          const obs = stepRes.data.observation;
          setMetrics({
            cash: obs.cash,
            accountValue: obs.account_value,
            positionShares: obs.position_shares,
            price: obs.tf_1m.ohlcv.close,
            regime: obs.market_regime,
            pattern: obs.tf_1m.chart_pattern,
            rawObservation: obs
          });
          if (stepRes.data.reward) {
             addLog('system', `Step Reward: ${stepRes.data.reward.toFixed(4)}`);
          }
        } catch (e) {
          addLog('error', `Env step failed: ${e.message}`);
        }
      } else {
        // Update local metrics state slightly to simulate a step (mock mode)
        if (action.action_type === 'open_long') {
          const cost = action.amount_shares * metrics.price;
          setMetrics(m => ({ 
            ...m, 
            positionShares: m.positionShares + action.amount_shares, 
            cash: m.cash - cost,
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
          setMetrics(m => ({ 
            ...m, 
            cash: m.cash + (m.positionShares > 0 ? value : -value), 
            positionShares: 0,
            price: m.price * (1 + (Math.random() * 0.02 - 0.01))
          }));
        }
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
