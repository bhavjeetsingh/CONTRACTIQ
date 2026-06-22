"use client";

import { useState, useEffect, Suspense } from "react";
import { useRouter, useSearchParams } from "next/navigation";
import Link from "next/link";
import axios from "axios";
import { Lock, Mail, Loader2, Sparkles, AlertCircle } from "lucide-react";

function LoginForm() {
  const router = useRouter();
  const searchParams = useSearchParams();
  
  const [isSignUp, setIsSignUp] = useState(false);
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (searchParams.get("signup") === "true") {
      setIsSignUp(true);
    }
  }, [searchParams]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    const baseUrl = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8080";
    const endpoint = isSignUp ? `${baseUrl}/auth/register` : `${baseUrl}/auth/login`;

    try {
      if (isSignUp) {
        // Register
        await axios.post(endpoint, { email, password });
        // After signup, automatically login
        const loginRes = await axios.post(`${baseUrl}/auth/login`, { email, password });
        localStorage.setItem("contractiq_token", loginRes.data.access_token);
        localStorage.setItem("contractiq_user", email);
        router.push("/dashboard");
      } else {
        // Login
        const res = await axios.post(endpoint, { email, password });
        localStorage.setItem("contractiq_token", res.data.access_token);
        localStorage.setItem("contractiq_user", email);
        router.push("/dashboard");
      }
    } catch (err: any) {
      logError(err);
      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError(isSignUp ? "Registration failed. Check inputs." : "Incorrect email or password.");
      }
    } finally {
      setLoading(false);
    }
  };

  const logError = (err: any) => {
    console.error("Authentication error:", err);
  };

  return (
    <div className="flex-1 flex flex-col items-center justify-center min-h-screen bg-background px-6 relative">
      {/* Glow animations */}
      <div className="absolute top-1/4 left-1/3 w-80 h-80 bg-indigo-500/10 rounded-full blur-3xl -z-10" />
      <div className="absolute bottom-1/4 right-1/3 w-80 h-80 bg-purple-500/10 rounded-full blur-3xl -z-10" />

      {/* Brand logo */}
      <Link href="/" className="flex items-center space-x-2 mb-8">
        <div className="w-9 h-9 rounded-xl bg-primary flex items-center justify-center font-bold text-white shadow-lg shadow-indigo-500/30">
          C
        </div>
        <span className="text-2xl font-bold tracking-tight text-white">
          ContractIQ
        </span>
      </Link>

      {/* Login Card */}
      <div className="w-full max-w-md glass-panel p-8 rounded-2xl glow-indigo">
        <h2 className="text-2xl font-bold tracking-tight mb-2 text-white">
          {isSignUp ? "Create your account" : "Welcome back"}
        </h2>
        <p className="text-sm text-zinc-400 mb-6">
          {isSignUp ? "Enter your email to sign up for free" : "Enter your email and password to log in"}
        </p>

        {error && (
          <div className="mb-6 px-4 py-3 rounded-xl bg-rose-500/10 border border-rose-500/20 text-rose-400 text-sm flex items-center space-x-2">
            <AlertCircle className="w-5 h-5 flex-shrink-0" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
              Email Address
            </label>
            <div className="relative">
              <Mail className="absolute left-4 top-3.5 w-5 h-5 text-zinc-500" />
              <input
                type="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                placeholder="you@example.com"
                className="w-full bg-zinc-900 border border-zinc-800 focus:border-primary focus:ring-1 focus:ring-primary rounded-xl pl-12 pr-4 py-3 text-sm text-white placeholder-zinc-500 outline-none transition-all"
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold text-zinc-400 uppercase tracking-wider mb-2">
              Password
            </label>
            <div className="relative">
              <Lock className="absolute left-4 top-3.5 w-5 h-5 text-zinc-500" />
              <input
                type="password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full bg-zinc-900 border border-zinc-800 focus:border-primary focus:ring-1 focus:ring-primary rounded-xl pl-12 pr-4 py-3 text-sm text-white placeholder-zinc-500 outline-none transition-all"
              />
            </div>
          </div>

          <button
            type="submit"
            disabled={loading}
            className="w-full mt-6 py-3.5 rounded-xl bg-primary hover:bg-primary/90 disabled:opacity-50 text-white font-semibold flex items-center justify-center space-x-2 transition-all duration-200 shadow-md shadow-indigo-500/20"
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <>
                <Sparkles className="w-4.5 h-4.5" />
                <span>{isSignUp ? "Sign Up Free" : "Sign In"}</span>
              </>
            )}
          </button>
        </form>

        <div className="mt-6 text-center">
          <button
            onClick={() => {
              setError(null);
              setIsSignUp(!isSignUp);
            }}
            className="text-sm font-medium text-primary hover:underline"
          >
            {isSignUp ? "Already have an account? Sign In" : "Don't have an account? Sign Up"}
          </button>
        </div>
      </div>
    </div>
  );
}


export default function LoginPage() {
  return (
    <Suspense fallback={
      <div className="flex-1 flex flex-col items-center justify-center min-h-screen bg-background text-zinc-400">
        <Loader2 className="w-8 h-8 animate-spin text-primary mb-2" />
        <span>Loading authentication...</span>
      </div>
    }>
      <LoginForm />
    </Suspense>
  );
}
