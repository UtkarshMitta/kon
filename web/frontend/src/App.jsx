import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import {
  Send,
  Settings,
  Database,
  ShieldCheck,
  Zap,
  TrendingUp,
  History,
  RotateCcw,
  Cpu,
  ArrowRight
} from 'lucide-react';

const API_BASE = "http://localhost:8000";

const Markdown = ({ content }) => {
  if (!content) return null;

  // Simple regex-based markdown parser
  const parseMarkdown = (text) => {
    let html = text
      .replace(/^### (.*$)/gim, '<h3>$1</h3>')
      .replace(/^## (.*$)/gim, '<h2>$1</h2>')
      .replace(/^# (.*$)/gim, '<h1>$1</h1>')
      .replace(/\*\*(.*)\*\*/gim, '<strong>$1</strong>')
      .replace(/\*(.*)\*/gim, '<em>$1</em>')
      .replace(/```([\s\S]*?)```/gim, '<pre><code>$1</code></pre>')
      .replace(/`(.*?)`/gim, '<code>$1</code>')
      .replace(/^\s*-\s+(.*)$/gim, '<ul><li>$1</li></ul>')
      .replace(/<\/ul>\s*<ul>/gim, '') // merge consecutive lists
      .replace(/\n/gim, '<br />');

    return { __html: html };
  };

  return <div className="markdown-content" dangerouslySetInnerHTML={parseMarkdown(content)} />;
};

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [compareMode, setCompareMode] = useState(false);
  const [stats, setStats] = useState({
    total_turns: 0,
    actual_total_cost: 0,
    baseline_total_cost: 0,
    total_savings: 0,
    roi_percentage: 0,
    summarization_overhead: 0
  });
  const [currentModel, setCurrentModel] = useState(null);
  const [isSwapping, setIsSwapping] = useState(false);

  const chatEndRef = useRef(null);

  const scrollToBottom = () => {
    chatEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      const res = await axios.get(`${API_BASE}/api/stats`);
      setStats(res.data);
    } catch (e) {
      console.error("Failed to fetch stats", e);
    }
  };

  const handleSend = async (e) => {
    e.preventDefault();
    if (!input.trim() || isLoading) return;

    const userMsg = { role: 'user', content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const res = await axios.post(`${API_BASE}/api/query`, {
        query: input,
        compare: compareMode
      });

      const assistantMsg = {
        role: 'assistant',
        content: res.data.response,
        routing: res.data.routing,
        roi: res.data.roi,
        usage: res.data.usage,
        query: input // store for feedback
      };

      setMessages(prev => [...prev, assistantMsg]);
      setStats(res.data.session_roi);

      if (res.data.routing.model_name !== currentModel) {
        setIsSwapping(true);
        setCurrentModel(res.data.routing.model_name);
        setTimeout(() => setIsSwapping(false), 500);
      }
    } catch (error) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: "Error: Could not connect to the router backend."
      }]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = async () => {
    if (window.confirm("Are you sure you want to reset the session? All history will be cleared.")) {
      await axios.post(`${API_BASE}/api/reset`);
      setMessages([]);
      setCurrentModel(null);
      await fetchStats(); // Explicitly re-fetch fresh zeroed stats
    }
  };

  const submitFeedback = async (msgIndex, correctedTier) => {
    const msg = messages[msgIndex];
    if (!msg.routing) return;

    try {
      await axios.post(`${API_BASE}/api/feedback`, {
        query: msg.query,
        original_tier: msg.routing.model_tier,
        corrected_tier: correctedTier,
        confidence: msg.routing.confidence
      });

      const newMessages = [...messages];
      newMessages[msgIndex].feedbackSubmitted = correctedTier;
      setMessages(newMessages);
    } catch (e) {
      console.error("Feedback failed", e);
    }
  };

  return (
    <div className="dashboard-container">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="logo">
          <ShieldCheck size={28} />
          <span>Mistral Router</span>
        </div>

        <div className="stats-group">
          <div className="stat-card">
            <span className="stat-label">Total Savings (USD)</span>
            <span className="stat-value roi-ticker">${stats.total_savings.toFixed(6)}</span>
            <div className="stat-delta">
              <TrendingUp size={14} />
              <span>{stats.roi_percentage.toFixed(1)}% Efficiency</span>
            </div>
          </div>

          <div className="stat-card">
            <span className="stat-label">Baseline Cost (T3)</span>
            <span className="stat-value">${stats.baseline_total_cost.toFixed(6)}</span>
          </div>

          <div className="stat-card">
            <span className="stat-label">Summarization Extra</span>
            <span className="stat-value">${stats.summarization_overhead.toFixed(6)}</span>
          </div>

          {currentModel && (
            <div className={`model-card ${isSwapping ? 'model-swap' : ''}`}>
              <h3><Cpu size={14} /> Active Model</h3>
              <div className="model-name">{currentModel}</div>
            </div>
          )}
        </div>

        <div className="settings-group" style={{ marginTop: 'auto' }}>
          <div className="stat-card" style={{ cursor: 'pointer' }} onClick={() => setCompareMode(!compareMode)}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <span className="stat-label">Battle Mode</span>
              <div style={{
                width: '32px',
                height: '16px',
                background: compareMode ? 'var(--accent-blue)' : '#333',
                borderRadius: '8px',
                position: 'relative'
              }}>
                <div style={{
                  width: '12px',
                  height: '12px',
                  background: 'white',
                  borderRadius: '50%',
                  position: 'absolute',
                  top: '2px',
                  left: compareMode ? '18px' : '2px',
                  transition: '0.2s'
                }} />
              </div>
            </div>
            <p style={{ fontSize: '0.65rem', color: 'var(--text-muted)', marginTop: '4px' }}>
              {compareMode ? "Calling real baseline for proof" : "Using shadow math for speed"}
            </p>
          </div>

          <button className="stat-card" style={{
            width: '100%',
            textAlign: 'left',
            background: 'rgba(239, 68, 68, 0.1)',
            borderColor: 'rgba(239, 68, 68, 0.2)',
            color: 'var(--error)',
            marginTop: '1rem'
          }} onClick={handleReset}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '8.5rem' }}>
              <span className="stat-label" style={{ color: 'var(--error)' }}>Reset Session</span>
              <RotateCcw size={14} />
            </div>
          </button>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        <header style={{
          padding: '1.5rem 2rem',
          borderBottom: '1px solid var(--border-glass)',
          display: 'flex',
          justifyContent: 'space-between',
          alignItems: 'center',
          backdropFilter: 'blur(10px)',
          zIndex: 10
        }}>
          <div>
            <h2 style={{ fontSize: '1.25rem' }}>Agent Workspace</h2>
            <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Elastic Context & Dynamic Routing Active</p>
          </div>
          <div style={{ display: 'flex', gap: '1.5rem' }}>
            <div style={{ textAlign: 'right' }}>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Turns</p>
              <p style={{ fontWeight: 600 }}>{stats.total_turns}</p>
            </div>
            <div style={{ textAlign: 'right' }}>
              <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Session ROI</p>
              <p style={{ fontWeight: 600, color: 'var(--success)' }}>{stats.roi_percentage.toFixed(1)}%</p>
            </div>
          </div>
        </header>

        <div className="chat-history">
          {messages.length === 0 && (
            <div style={{
              display: 'flex',
              flexDirection: 'column',
              alignItems: 'center',
              justifyContent: 'center',
              height: '100%',
              opacity: 0.5
            }}>
              <Zap size={48} color="var(--accent-blue)" style={{ marginBottom: '1rem' }} />
              <p>Ready for your first query...</p>
            </div>
          )}

          {messages.map((msg, i) => (
            <div key={i} className={`message ${msg.role}`}>
              <Markdown content={msg.content} />

              {msg.routing && (
                <div className="meta-chips">
                  <div className="chip tier">
                    Tier {msg.routing.model_tier}
                  </div>
                  <div className="chip">
                    {msg.routing.model_name}
                  </div>
                  <div className="chip" style={{ color: 'var(--success)', borderColor: 'rgba(16, 185, 129, 0.2)' }}>
                    Saved ${(msg.roi.baseline_cost - msg.roi.actual_cost).toFixed(6)}
                  </div>
                </div>
              )}

              {msg.role === 'assistant' && msg.routing && (
                <div className="feedback-container">
                  {!msg.feedbackSubmitted ? (
                    <>
                      <span className="feedback-prompt-text">Are you satisfied with the answer?</span>
                      <div className="feedback-actions">
                        <button
                          className="feedback-btn"
                          onClick={() => submitFeedback(i, msg.routing.model_tier)}
                        >
                          Yes
                        </button>
                        <button
                          className="feedback-btn"
                          onClick={() => {
                            const nextTier = Math.min(msg.routing.model_tier + 1, 4);
                            submitFeedback(i, nextTier);
                          }}
                        >
                          No
                        </button>
                      </div>
                    </>
                  ) : (
                    <span className="feedback-prompt-text" style={{ color: 'var(--success)', opacity: 0.7 }}>
                      Feedback Recorded
                    </span>
                  )}
                </div>
              )}
            </div>
          ))}
          <div ref={chatEndRef} />
        </div>

        <div className="input-container">
          <form className="input-box" onSubmit={handleSend}>
            <Database size={20} color="var(--text-muted)" />
            <input
              type="text"
              placeholder="Type your query to route..."
              value={input}
              onChange={(e) => setInput(e.target.value)}
              disabled={isLoading}
            />
            <button type="submit" className="send-btn" disabled={isLoading}>
              {isLoading ? <RotateCcw size={18} className="spin" /> : <Send size={18} />}
            </button>
          </form>
        </div>
      </main>

      <style dangerouslySetInnerHTML={{
        __html: `
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
      `}} />
    </div>
  );
}

export default App;
