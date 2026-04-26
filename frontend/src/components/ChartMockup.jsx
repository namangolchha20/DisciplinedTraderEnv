import React from 'react';
import { LineChart, BarChart2 } from 'lucide-react';

const ChartMockup = ({ metrics }) => {
  return (
    <div className="glass-panel" style={{ height: '300px', display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <LineChart size={20} color="var(--accent-blue)" />
          <h2 style={{ fontSize: '1.1rem', margin: 0 }}>Market Stream</h2>
        </div>
        <div style={{ display: 'flex', gap: '1rem', fontSize: '0.85rem' }}>
          <span style={{ color: 'var(--text-secondary)' }}>Regime: <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{metrics.regime}</span></span>
          <span style={{ color: 'var(--text-secondary)' }}>Pattern: <span style={{ color: 'var(--text-primary)', fontWeight: 600 }}>{metrics.pattern}</span></span>
        </div>
      </div>
      
      <div style={{ 
        flex: 1, 
        backgroundColor: 'rgba(0,0,0,0.2)', 
        borderRadius: '8px', 
        border: '1px solid var(--border-color)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        position: 'relative',
        overflow: 'hidden'
      }}>
        {/* CSS Mockup of a chart */}
        <div style={{ position: 'absolute', inset: 0, opacity: 0.1, backgroundImage: 'linear-gradient(var(--border-color) 1px, transparent 1px), linear-gradient(90deg, var(--border-color) 1px, transparent 1px)', backgroundSize: '20px 20px' }}></div>
        
        <div style={{ display: 'flex', alignItems: 'flex-end', gap: '10px', height: '80%', width: '90%', zIndex: 1 }}>
          {[...Array(20)].map((_, i) => {
            const isUp = Math.random() > 0.4;
            const height = 20 + Math.random() * 60;
            return (
              <div key={i} style={{ 
                flex: 1, 
                backgroundColor: isUp ? 'var(--accent-green)' : 'var(--accent-red)',
                height: `${height}%`,
                borderRadius: '2px 2px 0 0',
                opacity: i === 19 ? 1 : 0.6,
                transform: i === 19 ? 'scaleY(1.1)' : 'none',
                transition: 'all 0.3s ease'
              }}></div>
            );
          })}
        </div>
        
        <div style={{ position: 'absolute', right: '10px', top: '20px', padding: '0.5rem 1rem', backgroundColor: 'var(--accent-blue)', color: 'white', borderRadius: '4px', fontWeight: 'bold', boxShadow: '0 4px 12px rgba(49, 130, 206, 0.4)' }}>
          ${metrics.price.toFixed(2)}
        </div>
      </div>
    </div>
  );
};

export default ChartMockup;
