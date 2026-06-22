"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import axios from "axios";
import {
  Upload, FileText, Download, LogOut, MessageSquare, Columns,
  Loader2, CheckCircle, AlertTriangle, AlertCircle, Award, Check, Sparkles
} from "lucide-react";

export default function DashboardPage() {
  const router = useRouter();
  
  const [token, setToken] = useState<string | null>(null);
  const [user, setUser] = useState<string | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [uploadStatus, setUploadStatus] = useState<string>(""); // idle, uploading, processing, done, error
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [extractedData, setExtractedData] = useState<any>(null);
  const [isOcr, setIsOcr] = useState<boolean>(false);
  const [avgConfidence, setAvgConfidence] = useState<number>(0);
  
  // Razorpay purchase integration state
  const [subscriptionTier, setSubscriptionTier] = useState<string>("free");
  const [upgrading, setUpgrading] = useState(false);

  useEffect(() => {
    const t = localStorage.getItem("contractiq_token");
    const u = localStorage.getItem("contractiq_user");
    if (!t) {
      router.push("/login");
    } else {
      setToken(t);
      setUser(u);
      fetchUserProfile(t);
    }
  }, []);

  const fetchUserProfile = async (authToken: string) => {
    try {
      const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
      // We can query health or any profile endpoints.
      // For now we query a health check, but we can verify profile status locally or mock.
      setSubscriptionTier(localStorage.getItem("contractiq_tier") || "free");
    } catch (err) {
      console.error(err);
    }
  };

  const handleLogout = () => {
    localStorage.removeItem("contractiq_token");
    localStorage.removeItem("contractiq_user");
    localStorage.removeItem("contractiq_tier");
    router.push("/");
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
    }
  };

  const handleUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!file) return;

    setUploadStatus("uploading");
    setErrorMessage(null);
    setExtractedData(null);

    const formData = new FormData();
    formData.append("file", file);

    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

    try {
      // Set headers with token
      const config = {
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "multipart/form-data"
        }
      };

      setUploadStatus("processing");
      // Use /analyze/v2 route
      const res = await axios.post(`${baseUrl}/analyze/v2`, formData, config);
      
      const resData = res.data;
      if (resData.error) {
        setErrorMessage(resData.error);
        setUploadStatus("error");
        return;
      }

      setExtractedData(resData.key_terms || resData.extracted_terms || resData.analysis);
      setAvgConfidence(resData.avg_confidence || resData.confidence_scores?.avg || 0.85);
      setIsOcr(resData.ocr_info?.is_ocr || false);
      setUploadStatus("done");
      
      // Save session id to local storage for export recovery
      if (resData.session_id) {
        localStorage.setItem("contractiq_session_id", resData.session_id);
      }
    } catch (err: any) {
      console.error(err);
      setUploadStatus("error");
      if (err.response?.data?.detail) {
        setErrorMessage(err.response.data.detail);
      } else {
        setErrorMessage("Upload failed. Make sure your file size is under 10MB.");
      }
    }
  };

  const handleExport = async (format: "json" | "csv" | "pdf") => {
    const sessionId = localStorage.getItem("contractiq_session_id");
    if (!sessionId) {
      alert("No active session found. Please analyze a contract first.");
      return;
    }
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
    const url = `${baseUrl}/export/${sessionId}/${format}`;

    try {
      const config = {
        headers: {
          "Authorization": `Bearer ${token}`
        },
        responseType: (format === "json" ? "json" : "blob") as "json" | "blob"
      };

      const res = await axios.get(url, config);

      if (format === "json") {
        const jsonStr = JSON.stringify(res.data.export || res.data, null, 2);
        const blob = new Blob([jsonStr], { type: "application/json" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `contractiq_${sessionId}.json`;
        link.click();
      } else {
        const blob = new Blob([res.data], { type: format === "pdf" ? "application/pdf" : "text/csv" });
        const link = document.createElement("a");
        link.href = URL.createObjectURL(blob);
        link.download = `contractiq_${sessionId}.${format}`;
        link.click();
      }
    } catch (err) {
      console.error(err);
      alert("Failed to export. Verify session credentials.");
    }
  };

  const triggerRazorpayCheckout = async () => {
    setUpgrading(true);
    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
    try {
      // 1. Create order
      const orderRes = await axios.post(
        `${baseUrl}/billing/order`,
        { tier: "premium" },
        { headers: { Authorization: `Bearer ${token}` } }
      );
      
      const order = orderRes.data;
      
      // Load Razorpay Script dynamically if not loaded
      if (!(window as any).Razorpay) {
        const script = document.createElement("script");
        script.src = "https://checkout.razorpay.com/v1/checkout.js";
        script.async = true;
        document.body.appendChild(script);
        await new Promise((resolve) => (script.onload = resolve));
      }
      
      // 2. Open popup
      const options = {
        key: order.razorpay_key_id, 
        amount: order.amount,
        currency: order.currency,
        name: "ContractIQ Premium",
        description: "Unlimited contract pages & GPU OCR",
        order_id: order.id,
        handler: async function (response: any) {
          try {
            // Verify payment
            const verifyRes = await axios.post(
              `${baseUrl}/billing/verify`,
              {
                razorpay_order_id: response.razorpay_order_id,
                razorpay_payment_id: response.razorpay_payment_id,
                razorpay_signature: response.razorpay_signature
              },
              { headers: { Authorization: `Bearer ${token}` } }
            );
            
            if (verifyRes.data.status === "verified") {
              setSubscriptionTier("premium");
              localStorage.setItem("contractiq_tier", "premium");
              alert("Payment successful! Account upgraded to Premium Pro! 🎉");
            }
          } catch (err) {
            console.error(err);
            alert("Payment verification failed. Please contact support.");
          }
        },
        prefill: {
          email: user || "user@example.com"
        },
        theme: {
          color: "#6366f1"
        }
      };
      
      const rzp = new (window as any).Razorpay(options);
      rzp.open();
    } catch (err: any) {
      console.error(err);
      alert("Failed to initialize billing order: " + (err.response?.data?.detail || err.message));
    } finally {
      setUpgrading(false);
    }
  };

  const setError = (err: any) => {
    setErrorMessage(err);
  };

  return (
    <div className="flex-1 flex flex-col min-h-screen bg-background">
      {/* Top Navigation */}
      <header className="glass-panel px-6 py-4 flex items-center justify-between sticky top-0 z-40">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center font-bold text-white shadow-lg">
            C
          </div>
          <span className="text-xl font-bold tracking-tight text-white">
            ContractIQ Dashboard
          </span>
        </div>
        <div className="flex items-center space-x-4">
          {subscriptionTier === "premium" ? (
            <div className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 text-xs font-semibold">
              <Award className="w-3.5 h-3.5" />
              <span>Premium Pro</span>
            </div>
          ) : (
            <button
              onClick={triggerRazorpayCheckout}
              disabled={upgrading}
              className="flex items-center space-x-1.5 px-3 py-1 rounded-full bg-primary/10 border border-primary/20 hover:bg-primary/20 text-primary text-xs font-semibold transition-all"
            >
              {upgrading ? (
                <Loader2 className="w-3 h-3 animate-spin" />
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Upgrade to Pro</span>
                </>
              )}
            </button>
          )}
          <span className="text-sm text-zinc-400 font-medium">
            {user || "Guest"}
          </span>
          <button
            onClick={handleLogout}
            className="p-2 rounded-lg bg-zinc-900 border border-zinc-800 text-zinc-400 hover:text-white transition-colors"
            title="Sign Out"
          >
            <LogOut className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* Main Workspace Layout */}
      <div className="flex-1 grid lg:grid-cols-4 gap-8 p-6 max-w-7xl mx-auto w-full">
        {/* Sidebar Actions */}
        <aside className="lg:col-span-1 space-y-6">
          <div className="glass-panel p-6 rounded-2xl">
            <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4">
              Workspaces
            </h2>
            <nav className="space-y-2">
              <Link
                href="/dashboard"
                className="flex items-center space-x-3 px-4 py-3 rounded-xl bg-zinc-900 text-white font-medium border border-zinc-800"
              >
                <FileText className="w-5 h-5 text-primary" />
                <span>Term Extraction</span>
              </Link>
              <Link
                href="/chat"
                className="flex items-center space-x-3 px-4 py-3 rounded-xl bg-transparent hover:bg-zinc-900 text-zinc-400 hover:text-white font-medium transition-colors"
              >
                <MessageSquare className="w-5 h-5" />
                <span>Hybrid RAG Chat</span>
              </Link>
              <Link
                href="/compare"
                className="flex items-center space-x-3 px-4 py-3 rounded-xl bg-transparent hover:bg-zinc-900 text-zinc-400 hover:text-white font-medium transition-colors"
              >
                <Columns className="w-5 h-5" />
                <span>Doc Comparison</span>
              </Link>
            </nav>
          </div>

          {/* Upload Area */}
          <div className="glass-panel p-6 rounded-2xl">
            <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4">
              Analyze Document
            </h2>
            <form onSubmit={handleUpload} className="space-y-4">
              <div className="border border-dashed border-zinc-800 rounded-xl p-6 text-center hover:border-zinc-700 transition-colors cursor-pointer relative bg-zinc-900/50">
                <input
                  type="file"
                  onChange={handleFileChange}
                  accept=".pdf,.docx,.doc,.txt,.md,.png,.jpg,.jpeg"
                  className="absolute inset-0 opacity-0 cursor-pointer"
                />
                <Upload className="w-8 h-8 text-zinc-500 mx-auto mb-3" />
                <span className="block text-xs font-semibold text-zinc-300 mb-1">
                  {file ? file.name : "Select Contract file"}
                </span>
                <span className="text-[10px] text-zinc-500 block">
                  PDF, DOCX, TXT, Images up to 10MB
                </span>
              </div>

              <button
                type="submit"
                disabled={!file || uploadStatus === "uploading" || uploadStatus === "processing"}
                className="w-full py-3 rounded-xl bg-primary hover:bg-primary/90 disabled:opacity-50 text-white font-semibold text-sm flex items-center justify-center space-x-2 transition-all"
              >
                {uploadStatus === "uploading" || uploadStatus === "processing" ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Processing Pipeline...</span>
                  </>
                ) : (
                  <>
                    <Upload className="w-4 h-4" />
                    <span>Run AI Analysis</span>
                  </>
                )}
              </button>
            </form>
            
            {uploadStatus === "processing" && (
              <div className="mt-4 space-y-2.5">
                <div className="flex items-center justify-between text-[11px] text-zinc-400">
                  <span>Pipeline Stepper</span>
                  <span className="animate-pulse text-primary font-medium">Extracting...</span>
                </div>
                <div className="h-1 bg-zinc-900 rounded-full overflow-hidden">
                  <div className="h-full bg-primary w-2/3 animate-pulse" />
                </div>
              </div>
            )}
          </div>
        </aside>

        {/* Results Workspace */}
        <main className="lg:col-span-3 space-y-6">
          {errorMessage && (
            <div className="px-6 py-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm flex items-center space-x-3">
              <AlertCircle className="w-6 h-6 flex-shrink-0" />
              <div>
                <p className="font-semibold">Pipeline Execution Failed</p>
                <p className="text-xs mt-0.5">{errorMessage}</p>
              </div>
            </div>
          )}

          {uploadStatus === "idle" && !extractedData && (
            <div className="glass-panel py-24 rounded-2xl text-center">
              <FileText className="w-16 h-16 text-zinc-600 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-zinc-300 mb-1">
                No active contract analysis
              </h3>
              <p className="text-sm text-zinc-500 max-w-md mx-auto">
                Drag a document into the sidebar upload area and trigger the AI execution pipeline to view key extracted variables.
              </p>
            </div>
          )}

          {extractedData && (
            <div className="space-y-6">
              {/* Header metrics card */}
              <div className="glass-panel p-6 rounded-2xl flex flex-wrap items-center justify-between gap-6">
                <div>
                  <h3 className="text-xl font-bold text-white mb-1 flex items-center space-x-2">
                    <span>Analysis Complete</span>
                    {isOcr && (
                      <span className="px-2 py-0.5 rounded bg-blue-500/10 border border-blue-500/20 text-[10px] text-blue-400 font-semibold uppercase tracking-wider">
                        Scanned (OCR)
                      </span>
                    )}
                  </h3>
                  <p className="text-sm text-zinc-400">
                    Results extracted via LangGraph self-correcting routing model.
                  </p>
                </div>
                <div className="flex items-center space-x-8">
                  <div className="text-center">
                    <span className="block text-[10px] font-semibold text-zinc-500 uppercase tracking-wider mb-1">
                      Avg Confidence
                    </span>
                    <span className={`text-xl font-bold ${avgConfidence >= 0.8 ? "text-emerald-400" : "text-amber-400"}`}>
                      {(avgConfidence * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => handleExport("json")}
                      className="flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-xs text-zinc-300 border border-zinc-800 transition-colors"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>JSON</span>
                    </button>
                    <button
                      onClick={() => handleExport("csv")}
                      className="flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-zinc-900 hover:bg-zinc-800 text-xs text-zinc-300 border border-zinc-800 transition-colors"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>CSV</span>
                    </button>
                    <button
                      onClick={() => handleExport("pdf")}
                      className="flex items-center space-x-1.5 px-3 py-2 rounded-lg bg-primary hover:bg-primary/90 text-xs text-white transition-all shadow-sm"
                    >
                      <Download className="w-3.5 h-3.5" />
                      <span>PDF Report</span>
                    </button>
                  </div>
                </div>
              </div>

              {/* Extraction grid */}
              <div className="grid md:grid-cols-2 gap-6">
                {/* Core Parameters Table */}
                <div className="glass-panel p-6 rounded-2xl">
                  <h4 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4 flex items-center space-x-2">
                    <CheckCircle className="w-4 h-4 text-emerald-400" />
                    <span>Extracted Core Parameters</span>
                  </h4>
                  <div className="border border-zinc-850 rounded-xl overflow-hidden text-sm">
                    <div className="divide-y divide-zinc-850 bg-zinc-900/30">
                      <div className="grid grid-cols-3 p-3">
                        <span className="font-semibold text-zinc-400">Parameter</span>
                        <span className="col-span-2 font-semibold text-zinc-400">Value</span>
                      </div>
                      <div className="grid grid-cols-3 p-3 text-zinc-300">
                        <span>Parties</span>
                        <span className="col-span-2 font-medium text-white">
                          {extractedData.parties
                            ? extractedData.parties.map((p: any) => `${p.name} (${p.role})`).join(", ")
                            : "Not found"}
                        </span>
                      </div>
                      <div className="grid grid-cols-3 p-3 text-zinc-300">
                        <span>Effective Date</span>
                        <span className="col-span-2 text-white">{extractedData.effective_date || "Not found"}</span>
                      </div>
                      <div className="grid grid-cols-3 p-3 text-zinc-300">
                        <span>Expiration Date</span>
                        <span className="col-span-2 text-white">{extractedData.expiration_date || "Not found"}</span>
                      </div>
                      <div className="grid grid-cols-3 p-3 text-zinc-300">
                        <span>Auto Renewal</span>
                        <span className="col-span-2 text-white">
                          {extractedData.auto_renewal !== undefined
                            ? extractedData.auto_renewal ? "Yes" : "No"
                            : "Not found"}
                        </span>
                      </div>
                      <div className="grid grid-cols-3 p-3 text-zinc-300">
                        <span>Governing Law</span>
                        <span className="col-span-2 text-white">{extractedData.governing_law || "Not found"}</span>
                      </div>
                      <div className="grid grid-cols-3 p-3 text-zinc-300">
                        <span>Liability Cap</span>
                        <span className="col-span-2 text-white">{extractedData.liability_cap || "Not found"}</span>
                      </div>
                      <div className="grid grid-cols-3 p-3 text-zinc-300">
                        <span>Confidentiality</span>
                        <span className="col-span-2 text-white">{extractedData.confidentiality_period || "Not found"}</span>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Risks Panel */}
                <div className="glass-panel p-6 rounded-2xl flex flex-col">
                  <h4 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4 flex items-center space-x-2">
                    <AlertTriangle className="w-4 h-4 text-amber-400" />
                    <span>Risk Flags & Legal Risks</span>
                  </h4>
                  <div className="space-y-3 overflow-y-auto max-h-[300px] flex-1">
                    {extractedData.risk_flags && extractedData.risk_flags.length > 0 ? (
                      extractedData.risk_flags.map((risk: any, i: number) => (
                        <div
                          key={i}
                          className="p-4 rounded-xl border bg-zinc-900/30 border-zinc-800 hover:border-zinc-700 transition-colors"
                        >
                          <div className="flex items-center justify-between mb-2">
                            <span className="text-xs font-bold uppercase tracking-wider text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded">
                              {risk.risk_level} Risk
                            </span>
                            {risk.page_reference && (
                              <span className="text-[10px] text-zinc-500">
                                Page {risk.page_reference}
                              </span>
                            )}
                          </div>
                          <p className="text-sm font-semibold text-white mb-1">
                            {risk.clause}
                          </p>
                          <p className="text-xs text-zinc-400 leading-relaxed mb-2">
                            {risk.plain_english}
                          </p>
                          <p className="text-[11px] text-primary italic">
                            Recommendation: {risk.recommendation}
                          </p>
                        </div>
                      ))
                    ) : (
                      <div className="p-8 text-center bg-zinc-900/20 border border-zinc-850 rounded-xl">
                        <Check className="w-8 h-8 text-emerald-400 mx-auto mb-2" />
                        <span className="text-xs text-zinc-400 font-medium">
                          No significant risk flags detected in this contract.
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Obligations list */}
              <div className="glass-panel p-6 rounded-2xl">
                <h4 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4 flex items-center space-x-2">
                  <FileText className="w-4 h-4 text-primary" />
                  <span>Key Obligations & Payment Terms</span>
                </h4>
                <div className="grid md:grid-cols-2 gap-6">
                  {/* Obligations */}
                  <div className="space-y-3">
                    <span className="block text-xs font-bold text-zinc-400 uppercase tracking-wide">
                      Active Commitments
                    </span>
                    <div className="space-y-2 max-h-[220px] overflow-y-auto">
                      {extractedData.obligations && extractedData.obligations.length > 0 ? (
                        extractedData.obligations.map((o: any, idx: number) => (
                          <div
                            key={idx}
                            className="p-3 bg-zinc-900/30 border border-zinc-800 rounded-xl text-xs"
                          >
                            <p className="font-semibold text-white mb-1">
                              {o.party}: {o.description}
                            </p>
                            <p className="text-zinc-500">
                              Deadline: <span className="text-zinc-300">{o.deadline}</span>
                            </p>
                          </div>
                        ))
                      ) : (
                        <p className="text-xs text-zinc-500">No specific commitments found.</p>
                      )}
                    </div>
                  </div>

                  {/* Payments */}
                  <div className="space-y-3">
                    <span className="block text-xs font-bold text-zinc-400 uppercase tracking-wide">
                      Payment Schedule
                    </span>
                    <div className="space-y-2 max-h-[220px] overflow-y-auto">
                      {extractedData.payment_terms && extractedData.payment_terms.length > 0 ? (
                        extractedData.payment_terms.map((p: any, idx: number) => (
                          <div
                            key={idx}
                            className="p-3 bg-zinc-900/30 border border-zinc-800 rounded-xl text-xs"
                          >
                            <p className="font-semibold text-white mb-1">
                              Amount: {p.amount} {p.currency}
                            </p>
                            <p className="text-zinc-400">
                              Frequency: <span className="text-white">{p.frequency}</span> | Due Date: <span className="text-white">{p.due_date}</span>
                            </p>
                          </div>
                        ))
                      ) : (
                        <p className="text-xs text-zinc-500">No payment terms detected.</p>
                      )}
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
