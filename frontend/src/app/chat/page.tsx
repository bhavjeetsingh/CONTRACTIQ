"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import axios from "axios";
import {
  Send, MessageSquare, FileText, Columns, ArrowLeft, Loader2,
  Trash2, Plus, Info, Globe, ShieldAlert
} from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
}

export default function ChatPage() {
  const router = useRouter();

  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<string | null>(null);
  const [sessionId, setSessionId] = useState<string | null>(null);
  
  // Chat state
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [sources, setSources] = useState<any[]>([]);
  
  // Index build state
  const [files, setFiles] = useState<File[]>([]);
  const [indexing, setIndexing] = useState(false);
  const [indexedFiles, setIndexedFiles] = useState<string[]>([]);

  useEffect(() => {
    const t = localStorage.getItem("contractiq_token");
    const u = localStorage.getItem("contractiq_user");
    const s = localStorage.getItem("contractiq_session_id");
    if (!t) {
      router.push("/login");
    } else {
      setToken(t);
      setUser(u);
      setSessionId(s);
      if (s) {
        loadChatHistory(t, s);
      }
    }
  }, []);

  const loadChatHistory = async (authToken: string, sessId: string) => {
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
    try {
      // In ContractIQ RAG, the query endpoint returns conversation history.
      // We can also query history directly if cached.
      // For simplicity, we initialize with a welcoming message.
      setMessages([
        { role: "assistant", content: "Hello! I am your contract intelligence copilot. Ask me any questions about the documents you indexed (e.g. 'What is the termination period?', 'Is there a liability limit?')." }
      ]);
    } catch (err) {
      console.error(err);
    }
  };

  const handleIndexFiles = async (e: React.FormEvent) => {
    e.preventDefault();
    if (files.length === 0) return;

    setIndexing(true);
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
    const formData = new FormData();
    files.forEach(f => formData.append("files", f));
    formData.append("use_session_dirs", "true");
    if (sessionId) {
      formData.append("session_id", sessionId);
    }

    try {
      const res = await axios.post(`${baseUrl}/chat/index`, formData, {
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "multipart/form-data"
        }
      });

      const newSessionId = res.data.session_id;
      setSessionId(newSessionId);
      localStorage.setItem("contractiq_session_id", newSessionId);
      setIndexedFiles(files.map(f => f.name));
      setFiles([]);
      setMessages([
        { role: "assistant", content: `Successfully indexed ${res.data.files_indexed} documents! You can now ask questions referencing these files.` }
      ]);
    } catch (err) {
      console.error(err);
      alert("Failed to build index. Please try again.");
    } finally {
      setIndexing(false);
    }
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!input.trim() || !sessionId) return;

    const userMessage = input.trim();
    setInput("");
    setMessages(prev => [...prev, { role: "user", content: userMessage }]);
    setLoading(true);

    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
    const formData = new FormData();
    formData.append("question", userMessage);
    formData.append("session_id", sessionId);
    formData.append("use_session_dirs", "true");

    try {
      const res = await axios.post(`${baseUrl}/chat/query`, formData, {
        headers: {
          "Authorization": `Bearer ${token}`
        }
      });

      const answer = res.data.answer;
      setMessages(prev => [...prev, { role: "assistant", content: answer }]);
      
      // Load sources if returned
      if (res.data.source_documents) {
        setSources(res.data.source_documents);
      }
    } catch (err) {
      console.error(err);
      setMessages(prev => [...prev, { role: "assistant", content: "Failed to query RAG model. Verify server connection." }]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearChat = () => {
    setMessages([
      { role: "assistant", content: "Chat history cleared. Ask me anything about the indexed contracts." }
    ]);
    setSources([]);
  };

  return (
    <div className="flex-1 flex flex-col min-h-screen bg-background">
      {/* Top Header */}
      <header className="glass-panel px-6 py-4 flex items-center justify-between sticky top-0 z-40">
        <div className="flex items-center space-x-3">
          <Link href="/dashboard" className="p-2 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white transition-colors">
            <ArrowLeft className="w-4 h-4" />
          </Link>
          <span className="text-xl font-bold tracking-tight text-white flex items-center space-x-2">
            <MessageSquare className="w-5 h-5 text-primary" />
            <span>Hybrid RAG Chat Workspace</span>
          </span>
        </div>
        <div className="flex items-center space-x-4">
          {sessionId && (
            <button
              onClick={handleClearChat}
              className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-xs text-zinc-400 hover:text-white border border-zinc-800 transition-colors"
            >
              <Trash2 className="w-3.5 h-3.5" />
              <span>Clear History</span>
            </button>
          )}
        </div>
      </header>

      {/* Main Container */}
      <div className="flex-1 grid lg:grid-cols-4 gap-8 p-6 max-w-7xl mx-auto w-full">
        {/* Left Side: Document Indexing Config */}
        <aside className="lg:col-span-1 space-y-6">
          <div className="glass-panel p-6 rounded-2xl">
            <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4">
              Indexing Workspace
            </h2>
            <form onSubmit={handleIndexFiles} className="space-y-4">
              <div className="border border-dashed border-zinc-800 rounded-xl p-6 text-center hover:border-zinc-700 transition-colors cursor-pointer relative bg-zinc-900/50">
                <input
                  type="file"
                  multiple
                  onChange={(e) => {
                    if (e.target.files) {
                      setFiles(Array.from(e.target.files));
                    }
                  }}
                  accept=".pdf,.docx,.doc,.txt,.md"
                  className="absolute inset-0 opacity-0 cursor-pointer"
                />
                <Plus className="w-8 h-8 text-zinc-500 mx-auto mb-3" />
                <span className="block text-xs font-semibold text-zinc-300 mb-1">
                  {files.length > 0 ? `${files.length} files selected` : "Add files to Index"}
                </span>
                <span className="text-[10px] text-zinc-500 block">
                  PDF, DOCX, TXT
                </span>
              </div>

              <button
                type="submit"
                disabled={files.length === 0 || indexing}
                className="w-full py-3 rounded-xl bg-primary hover:bg-primary/90 disabled:opacity-50 text-white font-semibold text-sm flex items-center justify-center space-x-2 transition-all"
              >
                {indexing ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Building Index...</span>
                  </>
                ) : (
                  <>
                    <Globe className="w-4 h-4" />
                    <span>Rebuild RAG Index</span>
                  </>
                )}
              </button>
            </form>
          </div>

          {/* Current Indexed Files */}
          <div className="glass-panel p-6 rounded-2xl">
            <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-3">
              Indexed Documents
            </h2>
            {indexedFiles.length > 0 ? (
              <div className="space-y-2 max-h-[200px] overflow-y-auto">
                {indexedFiles.map((name, i) => (
                  <div key={i} className="flex items-center space-x-2 text-xs text-zinc-300 bg-zinc-900/50 p-2.5 rounded-lg border border-zinc-850">
                    <FileText className="w-4 h-4 text-primary flex-shrink-0" />
                    <span className="truncate">{name}</span>
                  </div>
                ))}
              </div>
            ) : (
              <div className="flex items-center space-x-2 text-xs text-zinc-500 bg-zinc-900/10 p-4 border border-dashed border-zinc-850 rounded-xl">
                <Info className="w-4 h-4 flex-shrink-0" />
                <span>No active session documents indexed. Upload files to start.</span>
              </div>
            )}
          </div>
        </aside>

        {/* Right Side: Chat Dialog + Audit sources */}
        <main className="lg:col-span-3 grid md:grid-cols-3 gap-6">
          {/* Chat Workspace */}
          <div className="md:col-span-2 glass-panel rounded-2xl flex flex-col h-[75vh]">
            {/* Conversation Log */}
            <div className="flex-1 p-6 overflow-y-auto space-y-4">
              {messages.map((m, idx) => (
                <div
                  key={idx}
                  className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
                >
                  <div
                    className={`max-w-[85%] rounded-2xl p-4 text-sm leading-relaxed ${
                      m.role === "user"
                        ? "bg-primary text-white"
                        : "bg-zinc-900 border border-zinc-800 text-zinc-100"
                    }`}
                  >
                    {m.content}
                  </div>
                </div>
              ))}
              {loading && (
                <div className="flex justify-start">
                  <div className="bg-zinc-900 border border-zinc-800 rounded-2xl p-4 text-sm flex items-center space-x-2 text-zinc-400">
                    <Loader2 className="w-4 h-4 animate-spin text-primary" />
                    <span>Analyzing indexed vector chunks...</span>
                  </div>
                </div>
              )}
            </div>

            {/* Input Form */}
            <form onSubmit={handleSendMessage} className="p-4 border-t border-zinc-850 bg-zinc-900/30 flex items-center space-x-3 rounded-b-2xl">
              <input
                type="text"
                value={input}
                disabled={!sessionId || loading}
                onChange={(e) => setInput(e.target.value)}
                placeholder={sessionId ? "Ask a question about the contract terms..." : "Index documents first to ask questions"}
                className="flex-1 bg-zinc-900 border border-zinc-800 focus:border-primary focus:ring-1 focus:ring-primary rounded-xl px-4 py-3 text-sm text-white placeholder-zinc-500 outline-none transition-all disabled:opacity-50"
              />
              <button
                type="submit"
                disabled={!input.trim() || loading || !sessionId}
                className="p-3 rounded-xl bg-primary hover:bg-primary/90 text-white disabled:opacity-50 transition-colors shadow-md shadow-indigo-500/20"
              >
                <Send className="w-5 h-5" />
              </button>
            </form>
          </div>

          {/* Sources audit board */}
          <div className="md:col-span-1 glass-panel p-6 rounded-2xl flex flex-col h-[75vh]">
            <h3 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4 flex items-center space-x-2">
              <ShieldAlert className="w-4 h-4 text-primary" />
              <span>Context Citations</span>
            </h3>
            <div className="flex-1 overflow-y-auto space-y-3">
              {sources.length > 0 ? (
                sources.map((src, i) => (
                  <div key={i} className="p-3.5 bg-zinc-900/40 border border-zinc-800 rounded-xl">
                    <div className="flex items-center justify-between mb-2">
                      <span className="text-[10px] font-bold text-primary uppercase bg-primary/10 px-2 py-0.5 rounded">
                        Match {i + 1}
                      </span>
                      {src.metadata?.page && (
                        <span className="text-[10px] text-zinc-500">
                          Page {src.metadata.page}
                        </span>
                      )}
                    </div>
                    <p className="text-[11px] text-zinc-300 leading-relaxed italic">
                      "{src.page_content?.slice(0, 200)}..."
                    </p>
                  </div>
                ))
              ) : (
                <div className="text-center py-20 text-zinc-600">
                  <Info className="w-8 h-8 mx-auto mb-2 text-zinc-700" />
                  <p className="text-xs">No citations retrieved yet.</p>
                </div>
              )}
            </div>
          </div>
        </main>
      </div>
    </div>
  );
}
