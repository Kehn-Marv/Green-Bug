import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Navigation from './components/Navigation';
import AnalyzeSection from './components/AnalyzeSection';
import BatchAnalyzeSection from './components/BatchAnalyzeSection';
import HealthSection from './components/HealthSection';
import StatsSection from './components/StatsSection';
import FamiliesSection from './components/FamiliesSection';
import LearningSection from './components/LearningSection';
import { getHealth } from './services/api';

type ActiveSection = 'analyze' | 'batch' | 'health' | 'stats' | 'families' | 'learning';

function App() {
  const [activeSection, setActiveSection] = useState<ActiveSection>('analyze');
  const [systemStatus, setSystemStatus] = useState<'healthy' | 'unhealthy' | 'unknown'>('unknown');

  // Check system health on app load
  useEffect(() => {
    const checkHealth = async () => {
      try {
        const health = await getHealth();
        setSystemStatus(health.status === 'healthy' ? 'healthy' : 'unhealthy');
      } catch {
        setSystemStatus('unhealthy');
      }
    };

    checkHealth();
    // Check health every 5 minutes
    const interval = setInterval(checkHealth, 300000);
    return () => clearInterval(interval);
  }, []);

  const renderActiveSection = () => {
    switch (activeSection) {
      case 'analyze':
        return <AnalyzeSection />;
      case 'batch':
        return <BatchAnalyzeSection />;
      case 'health':
        return <HealthSection />;
      case 'stats':
        return <StatsSection />;
      case 'families':
        return <FamiliesSection />;
      case 'learning':
        return <LearningSection />;
      default:
        return <AnalyzeSection />;
    }
  };

  return (
    <div className="min-h-screen bg-gray-50">
      <Header systemStatus={systemStatus} />
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        <Navigation activeSection={activeSection} onSectionChange={setActiveSection} />
        <div className="mt-8">
          {renderActiveSection()}
        </div>
      </div>
    </div>
  );
}

export default App;