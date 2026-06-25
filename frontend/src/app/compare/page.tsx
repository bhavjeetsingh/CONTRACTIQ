"use client";

import { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import Link from "next/link";
import axios from "axios";
import {
  Columns, Upload, FileText, ArrowLeft, Loader2, CheckCircle,
  AlertTriangle, AlertCircle, FilePlus
} from "lucide-react";

interface CompareRow {
  clause: string;
  reference_value: string;
  actual_value: string;
  difference_type: string;
  severity: "match" | "minor_difference" | "major_difference" | "critical_risk";
}

export default function ComparePage() {
  const router = useRouter();

  const [token, setToken] = useState<string | null>(null);
  const [refFile, setRefFile] = useState<File | null>(null);
  const [actFile, setActFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [rows, setRows] = useState<CompareRow[]>([]);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const t = localStorage.getItem("contractiq_token");
    if (!t) {
      router.push("/login");
    } else {
      setToken(t);
    }
  }, []);

  const handleCompare = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!refFile || !actFile) return;

    setLoading(true);
    setError(null);
    setRows([]);

    const formData = new FormData();
    formData.append("reference", refFile);
    formData.append("actual", actFile);

    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";

    try {
      const res = await axios.post(`${baseUrl}/compare`, formData, {
        headers: {
          "Authorization": `Bearer ${token}`,
          "Content-Type": "multipart/form-data"
        }
      });

      // Response contains {"rows": [...], "session_id": "..."}
      setRows(res.data.rows || []);
    } catch (err: any) {
      console.error(err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError("Failed to run comparison. Ensure both files are valid documents.");
      }
    } finally {
      setLoading(false);
    }
  };

  const getSeverityBadgeColor = (severity: string) => {
    switch (severity) {
      case "match":
        return "bg-emerald-500/10 border-emerald-500/20 text-emerald-400";
      case "minor_difference":
        return "bg-blue-500/10 border-blue-500/20 text-blue-400";
      case "major_difference":
        return "bg-amber-500/10 border-amber-500/20 text-amber-400";
      case "critical_risk":
        return "bg-rose-500/10 border-rose-500/20 text-rose-400";
      default:
        return "bg-zinc-500/10 border-zinc-500/20 text-zinc-400";
    }
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
            <Columns className="w-5 h-5 text-primary" />
            <span>Document Version Comparison Workspace</span>
          </span>
        </div>
      </header>

      {/* Layout Split */}
      <div className="flex-1 grid lg:grid-cols-4 gap-8 p-6 max-w-7xl mx-auto w-full">
        {/* Left Upload Panel */}
        <aside className="lg:col-span-1 space-y-6">
          <div className="glass-panel p-6 rounded-2xl">
            <h2 className="text-sm font-semibold text-zinc-400 uppercase tracking-wider mb-4">
              Select Versions
            </h2>
            <form onSubmit={handleCompare} className="space-y-4">
              {/* Reference upload */}
              <div>
                <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                  Reference File (Standard template)
                </label>
                <div className="border border-dashed border-zinc-800 rounded-xl p-4 text-center hover:border-zinc-700 transition-colors cursor-pointer relative bg-zinc-900/30">
                  <input
                    type="file"
                    onChange={(e) => e.target.files && setRefFile(e.target.files[0])}
                    accept=".pdf,.docx,.doc,.txt,.md"
                    className="absolute inset-0 opacity-0 cursor-pointer"
                  />
                  <FilePlus className="w-6 h-6 text-zinc-500 mx-auto mb-2" />
                  <span className="block text-[11px] font-semibold text-zinc-300 truncate">
                    {refFile ? refFile.name : "Select Reference"}
                  </span>
                </div>
              </div>

              {/* Actual upload */}
              <div>
                <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
                  Actual File (Signed/Redlined version)
                </label>
                <div className="border border-dashed border-zinc-800 rounded-xl p-4 text-center hover:border-zinc-700 transition-colors cursor-pointer relative bg-zinc-900/30">
                  <input
                    type="file"
                    onChange={(e) => e.target.files && setActFile(e.target.files[0])}
                    accept=".pdf,.docx,.doc,.txt,.md"
                    className="absolute inset-0 opacity-0 cursor-pointer"
                  />
                  <FilePlus className="w-6 h-6 text-zinc-500 mx-auto mb-2" />
                  <span className="block text-[11px] font-semibold text-zinc-300 truncate">
                    {actFile ? actFile.name : "Select Actual"}
                  </span>
                </div>
              </div>

              <button
                type="submit"
                disabled={!refFile || !actFile || loading}
                className="w-full mt-4 py-3 rounded-xl bg-primary hover:bg-primary/90 disabled:opacity-50 text-white font-semibold text-sm flex items-center justify-center space-x-2 transition-all"
              >
                {loading ? (
                  <>
                    <Loader2 className="w-4 h-4 animate-spin" />
                    <span>Analyzing differences...</span>
                  </>
                ) : (
                  <>
                    <Columns className="w-4 h-4" />
                    <span>Run Comparison</span>
                  </>
                )}
              </button>
            </form>
          </div>
        </aside>

        {/* Right Comparison View */}
        <main className="lg:col-span-3 space-y-6">
          {error && (
            <div className="px-6 py-4 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm flex items-center space-x-3">
              <AlertCircle className="w-6 h-6 flex-shrink-0" />
              <span>{error}</span>
            </div>
          )}

          {rows.length === 0 && !loading && (
            <div className="glass-panel py-24 rounded-2xl text-center">
              <Columns className="w-16 h-16 text-zinc-650 mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-zinc-300 mb-1">
                No active contract comparison
              </h3>
              <p className="text-sm text-zinc-500 max-w-md mx-auto">
                Select your reference document and actual document in the upload panel to see an automated audit of changes and deviations.
              </p>
            </div>
          )}

          {loading && (
            <div className="glass-panel py-24 rounded-2xl text-center">
              <Loader2 className="w-10 h-10 animate-spin text-primary mx-auto mb-4" />
              <h3 className="text-lg font-semibold text-white mb-1">
                Analyzing document versions
              </h3>
              <p className="text-sm text-zinc-400 max-w-sm mx-auto">
                Parsing paragraph lines and resolving clauses. This might take 10-15 seconds.
              </p>
            </div>
          )}

          {rows.length > 0 && (
            <div className="glass-panel p-6 rounded-2xl">
              <h3 className="text-lg font-bold text-white mb-4 flex items-center space-x-2">
                <CheckCircle className="w-5 h-5 text-emerald-400" />
                <span>Automated Audit Log</span>
              </h3>
              <div className="border border-zinc-850 rounded-xl overflow-hidden text-sm">
                <div className="divide-y divide-zinc-850 bg-zinc-900/10">
                  {/* Table Header */}
                  <div className="grid grid-cols-12 p-4 font-semibold text-zinc-400 bg-zinc-900/30">
                    <div className="col-span-2">Page</div>
                    <div className="col-span-10">Summary of Changes & Deviations</div>
                  </div>
                  {/* Table Body */}
                  {rows.map((row: any, idx) => {
                    const pageVal = row.Page || row.page || "N/A";
                    const changesVal = row.Changes || row.changes || row.description || "";
                    return (
                      <div key={idx} className="grid grid-cols-12 p-4 items-start text-zinc-300 hover:bg-zinc-900/20 transition-colors border-t border-zinc-800">
                        <div className="col-span-2 font-bold text-primary">Page {pageVal}</div>
                        <div className="col-span-10 text-sm whitespace-pre-wrap leading-relaxed text-zinc-300">{changesVal}</div>
                      </div>
                    );
                  })}
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    </div>
  );
}
