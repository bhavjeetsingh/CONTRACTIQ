"use client";

import Link from "next/link";
import { useState, useEffect } from "react";
import { ArrowRight, CheckCircle2, ShieldAlert, Cpu, Sparkles, Database, FileText } from "lucide-react";

export default function LandingPage() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);

  useEffect(() => {
    // Simple mock verification for hydration, can hook to actual auth storage
    const token = localStorage.getItem("contractiq_token");
    if (token) {
      setIsAuthenticated(true);
    }
  }, []);

  return (
    <div className="flex flex-col min-h-screen bg-background text-foreground overflow-hidden">
      {/* Background glow animations */}
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-primary/10 rounded-full blur-3xl -z-10 animate-pulse" />
      <div className="absolute bottom-10 right-1/4 w-96 h-96 bg-indigo-500/10 rounded-full blur-3xl -z-10" />

      {/* Navbar */}
      <nav className="glass-panel sticky top-0 z-50 px-6 py-4 flex items-center justify-between">
        <div className="flex items-center space-x-2">
          <div className="w-8 h-8 rounded-lg bg-primary flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/30">
            C
          </div>
          <span className="text-xl font-bold tracking-tight bg-gradient-to-r from-white to-zinc-400 bg-clip-text text-transparent">
            ContractIQ
          </span>
        </div>
        <div className="flex items-center space-x-6">
          <Link href="#features" className="text-sm font-medium text-zinc-400 hover:text-white transition-colors">
            Features
          </Link>
          <Link href="#pricing" className="text-sm font-medium text-zinc-400 hover:text-white transition-colors">
            Pricing
          </Link>
          {isAuthenticated ? (
            <Link
              href="/dashboard"
              className="px-4 py-2 text-sm font-medium rounded-lg bg-primary hover:bg-primary/90 text-white transition-all duration-200 glow-indigo"
            >
              Go to Dashboard
            </Link>
          ) : (
            <>
              <Link href="/login" className="text-sm font-medium text-zinc-300 hover:text-white transition-colors">
                Sign In
              </Link>
              <Link
                href="/login?signup=true"
                className="px-4 py-2 text-sm font-medium rounded-lg bg-primary hover:bg-primary/90 text-white transition-all duration-200 glow-indigo"
              >
                Sign Up
              </Link>
            </>
          )}
        </div>
      </nav>

      {/* Hero Section */}
      <section className="flex-1 flex flex-col items-center justify-center text-center px-6 py-20 md:py-32">
        <div className="inline-flex items-center space-x-2 px-3 py-1.5 rounded-full bg-zinc-900 border border-zinc-800 text-xs font-semibold text-primary mb-8 tracking-wide">
          <Sparkles className="w-3.5 h-3.5" />
          <span>LangGraph Self-Correcting Agentic Pipelines</span>
        </div>
        <h1 className="text-5xl md:text-7xl font-extrabold tracking-tight max-w-4xl leading-tight mb-8">
          AI Contract Intelligence
          <span className="block mt-2 bg-gradient-to-r from-primary via-purple-500 to-indigo-400 bg-clip-text text-transparent">
            For Builders & Legal Teams
          </span>
        </h1>
        <p className="text-lg md:text-xl text-zinc-400 max-w-2xl leading-relaxed mb-12">
          Upload any contract (digital or scanned PDF) to extract structured key terms, flag liabilities, and chat with your legal documents using advanced hybrid RAG.
        </p>
        <div className="flex flex-col sm:flex-row items-center justify-center gap-4">
          <Link
            href="/dashboard"
            className="w-full sm:w-auto px-8 py-4 rounded-xl bg-primary hover:bg-primary/90 text-white font-semibold transition-all duration-200 flex items-center justify-center space-x-2 group shadow-lg shadow-indigo-500/20"
          >
            <span>Analyze Contract Free</span>
            <ArrowRight className="w-5 h-5 group-hover:translate-x-1 transition-transform" />
          </Link>
          <Link
            href="#pricing"
            className="w-full sm:w-auto px-8 py-4 rounded-xl bg-zinc-900 hover:bg-zinc-800 text-zinc-300 font-semibold border border-zinc-800 transition-colors"
          >
            View Pricing Tiers
          </Link>
        </div>
      </section>

      {/* Features Section */}
      <section id="features" className="py-24 px-6 bg-zinc-950 border-y border-zinc-900">
        <div className="max-w-6xl mx-auto">
          <div className="text-center mb-16">
            <h2 className="text-3xl md:text-4xl font-bold tracking-tight mb-4">
              Designed For High Precision Legal Analysis
            </h2>
            <p className="text-zinc-400 max-w-2xl mx-auto">
              Linear RAG chains hallucinate. ContractIQ uses agentic routing and structured schemas to guarantee correctness.
            </p>
          </div>
          <div className="grid md:grid-cols-3 gap-8">
            <div className="glass-card p-8 rounded-2xl flex flex-col items-start hover:border-zinc-700 transition-colors duration-200">
              <div className="w-12 h-12 rounded-xl bg-blue-500/10 flex items-center justify-center text-blue-400 mb-6">
                <Cpu className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Self-Correcting LangGraph</h3>
              <p className="text-zinc-400 text-sm leading-relaxed">
                If the model extracts information below confidence thresholds, the validation node initiates automatic retry loops with tailored hints.
              </p>
            </div>
            <div className="glass-card p-8 rounded-2xl flex flex-col items-start hover:border-zinc-700 transition-colors duration-200">
              <div className="w-12 h-12 rounded-xl bg-purple-500/10 flex items-center justify-center text-purple-400 mb-6">
                <Database className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Hybrid RAG & OCR Fallback</h3>
              <p className="text-zinc-400 text-sm leading-relaxed">
                Seamlessly handles image uploads and scanned documents using OCR fallback. Implements Reciprocal Rank Fusion (RRF) for retrieval.
              </p>
            </div>
            <div className="glass-card p-8 rounded-2xl flex flex-col items-start hover:border-zinc-700 transition-colors duration-200">
              <div className="w-12 h-12 rounded-xl bg-emerald-500/10 flex items-center justify-center text-emerald-400 mb-6">
                <FileText className="w-6 h-6" />
              </div>
              <h3 className="text-xl font-semibold mb-3">Cloud Storage & Exports</h3>
              <p className="text-zinc-400 text-sm leading-relaxed">
                Uploaded contracts sync directly to Supabase cloud storage. Export structured profiles to professional JSON, CSV, or PDF summaries.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Pricing Section */}
      <section id="pricing" className="py-24 px-6 max-w-6xl mx-auto">
        <div className="text-center mb-16">
          <h2 className="text-3xl md:text-5xl font-bold tracking-tight mb-4">
            Flexible Pricing For Scale
          </h2>
          <p className="text-zinc-400 max-w-xl mx-auto">
            Choose a tier that fits your legal analysis demands. Pay with cards, UPI, or net banking via Razorpay.
          </p>
        </div>
        <div className="grid md:grid-cols-2 gap-8 max-w-4xl mx-auto items-stretch">
          {/* Free Tier */}
          <div className="glass-card p-8 rounded-2xl flex flex-col border border-zinc-800 hover:border-zinc-700 transition-all duration-200">
            <h3 className="text-lg font-semibold text-zinc-400 mb-2">Free Plan</h3>
            <div className="flex items-baseline mb-6">
              <span className="text-4xl font-extrabold tracking-tight">₹0</span>
              <span className="text-zinc-500 ml-2">/ month</span>
            </div>
            <p className="text-zinc-400 text-sm mb-8">
              Perfect for freelancers, founders, and students looking to review core terms of simple agreements.
            </p>
            <ul className="space-y-4 mb-10 flex-1">
              <li className="flex items-center space-x-3 text-sm text-zinc-300">
                <CheckCircle2 className="w-5 h-5 text-zinc-500 flex-shrink-0" />
                <span>50 pages processed per month</span>
              </li>
              <li className="flex items-center space-x-3 text-sm text-zinc-300">
                <CheckCircle2 className="w-5 h-5 text-zinc-500 flex-shrink-0" />
                <span>Key-term extraction & validation</span>
              </li>
              <li className="flex items-center space-x-3 text-sm text-zinc-300">
                <CheckCircle2 className="w-5 h-5 text-zinc-500 flex-shrink-0" />
                <span>Basic export formats (JSON, CSV)</span>
              </li>
              <li className="flex items-center space-x-3 text-sm text-zinc-300">
                <CheckCircle2 className="w-5 h-5 text-zinc-500 flex-shrink-0" />
                <span>Hybrid RAG chat workspace</span>
              </li>
            </ul>
            <Link
              href="/dashboard"
              className="w-full py-3.5 rounded-xl bg-zinc-900 border border-zinc-800 hover:bg-zinc-850 text-center font-medium text-white transition-colors"
            >
              Get Started Free
            </Link>
          </div>

          {/* Premium Tier */}
          <div className="glass-card p-8 rounded-2xl flex flex-col relative border-2 border-primary glow-indigo transition-all duration-200">
            <div className="absolute -top-3 right-6 px-3 py-1 bg-primary text-white text-xs font-semibold rounded-full uppercase tracking-wider">
              Popular Choice
            </div>
            <h3 className="text-lg font-semibold text-primary mb-2">Premium Pro</h3>
            <div className="flex items-baseline mb-6">
              <span className="text-4xl font-extrabold tracking-tight">₹499</span>
              <span className="text-zinc-500 ml-2">/ month</span>
            </div>
            <p className="text-zinc-400 text-sm mb-8">
              Designed for professional agencies, growing startups, and builders who handle bulk NDAs, vendor MSAs, or scanned receipts.
            </p>
            <ul className="space-y-4 mb-10 flex-1">
              <li className="flex items-center space-x-3 text-sm text-zinc-300">
                <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0" />
                <span className="font-semibold text-white">Unlimited processed pages</span>
              </li>
              <li className="flex items-center space-x-3 text-sm text-zinc-300">
                <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0" />
                <span>Premium PDF customized reports</span>
              </li>
              <li className="flex items-center space-x-3 text-sm text-zinc-300">
                <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0" />
                <span>Priority queue processing & GPU OCR</span>
              </li>
              <li className="flex items-center space-x-3 text-sm text-zinc-300">
                <CheckCircle2 className="w-5 h-5 text-primary flex-shrink-0" />
                <span>Detailed governing law & non-compete scoring</span>
              </li>
            </ul>
            <Link
              href="/dashboard?upgrade=true"
              className="w-full py-3.5 rounded-xl bg-primary hover:bg-primary/90 text-center font-semibold text-white transition-all duration-200 shadow-md shadow-indigo-500/20"
            >
              Upgrade Now
            </Link>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6 bg-zinc-950 border-t border-zinc-900 text-center text-sm text-zinc-500">
        <p>© {new Date().getFullYear()} ContractIQ. Built for fast and secure contract workflows.</p>
      </footer>
    </div>
  );
}
