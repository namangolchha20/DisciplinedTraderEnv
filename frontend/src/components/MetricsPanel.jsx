import React from 'react';
import { Wallet, PieChart, TrendingUp, TrendingDown, Target } from 'lucide-react';

const MetricsPanel = ({ metrics }) => {
  const profitLoss = metrics.accountValue - 10000;
  const plPercentage = (profitLoss / 10000) * 100;
  
  const isProfit = profitLoss >= 0;

  return (
    <div className="glass-panel" style={{ height: '100%', display: 'flex', flexDirection: 'column', gap: '1.5rem' }}>
      <h2 style={{ fontSize: '1.25rem', borderBottom: '1px solid var(--border-color)', paddingBottom: '0.75rem', margin: 0 }}>
        Account Dashboard
      </h2>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '1rem' }}>
        <MetricCard 
          icon={<Wallet size={20} color="var(--accent-blue)" />}
          label="Account Value"
          value={`$${metrics.accountValue.toLocaleString(undefined, {minimumFractionDigits: 2})}`}
          highlight={isProfit ? 'var(--accent-green)' : 'var(--text-primary)'}
        />
        
        <MetricCard 
          icon={<PieChart size={20} color="var(--accent-purple)" />}
          label="Available Cash"
          value={`$${metrics.cash.toLocaleString(undefined, {minimumFractionDigits: 2})}`}
        />

        <MetricCard 
          icon={metrics.positionShares > 0 ? <TrendingUp size={20} color="var(--accent-green)" /> : (metrics.positionShares < 0 ? <TrendingDown size={20} color="var(--accent-red)" /> : <Target size={20} color="var(--text-secondary)" />)}
          label="Open Position"
          value={metrics.positionShares === 0 ? "Flat" : `${metrics.positionShares > 0 ? 'Long' : 'Short'} ${Math.abs(metrics.positionShares)} shares`}
        />
      </div>

      <div style={{ marginTop: 'auto' }}>
        <div style={{ padding: '1rem', backgroundColor: 'rgba(0,0,0,0.2)', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
          <div style={{ fontSize: '0.85rem', color: 'var(--text-secondary)', marginBottom: '0.5rem' }}>Total P&L</div>
          <div style={{ fontSize: '1.5rem', fontWeight: 600, color: isProfit ? 'var(--accent-green)' : 'var(--accent-red)' }}>
            {isProfit ? '+' : ''}{profitLoss.toLocaleString(undefined, {minimumFractionDigits: 2})} 
            <span style={{ fontSize: '1rem', marginLeft: '0.5rem' }}>
              ({isProfit ? '+' : ''}{plPercentage.toFixed(2)}%)
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};

const MetricCard = ({ icon, label, value, highlight }) => (
  <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', padding: '0.75rem', backgroundColor: 'rgba(255,255,255,0.03)', borderRadius: '8px' }}>
    <div style={{ padding: '0.5rem', backgroundColor: 'rgba(255,255,255,0.05)', borderRadius: '8px', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
      {icon}
    </div>
    <div>
      <div style={{ fontSize: '0.8rem', color: 'var(--text-secondary)' }}>{label}</div>
      <div style={{ fontSize: '1.1rem', fontWeight: 600, color: highlight || 'var(--text-primary)' }}>{value}</div>
    </div>
  </div>
);

export default MetricsPanel;
