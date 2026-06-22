@import "tailwindcss";

@import url("https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap");

:root {
  --background: #050508;
  --foreground: #f4f4f5;
  
  --card: rgba(13, 13, 18, 0.7);
  --card-foreground: #f4f4f5;
  
  --popover: #050508;
  --popover-foreground: #f4f4f5;
  
  --primary: #4f46e5;
  --primary-foreground: #ffffff;
  
  --secondary: #18181b;
  --secondary-foreground: #f4f4f5;
  
  --muted: #18181b;
  --muted-foreground: #a1a1aa;
  
  --accent: #27272a;
  --accent-foreground: #f4f4f5;
  
  --destructive: #ef4444;
  --destructive-foreground: #fafafa;
  
  --border: rgba(255, 255, 255, 0.08);
  --input: rgba(255, 255, 255, 0.05);
  --ring: #4f46e5;
  
  --radius: 1rem;
}

body {
  background: var(--background);
  color: var(--foreground);
  font-family: "Inter", sans-serif;
  background-image: 
    radial-gradient(circle at 50% 0%, rgba(99, 102, 241, 0.08) 0%, transparent 60%),
    radial-gradient(circle at 100% 60%, rgba(147, 51, 234, 0.03) 0%, transparent 40%),
    linear-gradient(rgba(255, 255, 255, 0.003) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255, 255, 255, 0.003) 1px, transparent 1px);
  background-size: 100% 100%, 100% 100%, 64px 64px, 64px 64px;
}

h1, h2, h3, h4, .font-display {
  font-family: "Outfit", sans-serif;
}

/* Glassmorphism custom classes */
.glass-panel {
  background: rgba(10, 10, 15, 0.7);
  backdrop-filter: blur(20px);
  border: 1px solid rgba(255, 255, 255, 0.08);
  box-shadow: 0 4px 30px rgba(0, 0, 0, 0.4);
}

.glass-card {
  background: rgba(15, 15, 25, 0.45);
  backdrop-filter: blur(14px);
  border: 1px solid rgba(255, 255, 255, 0.05);
  transition: all 0.4s cubic-bezier(0.16, 1, 0.3, 1);
}

.glass-card:hover {
  border-color: rgba(99, 102, 241, 0.25);
  background: rgba(18, 18, 30, 0.65);
  transform: translateY(-4px);
  box-shadow: 0 10px 40px -10px rgba(0, 0, 0, 0.7), 0 0 20px -5px rgba(99, 102, 241, 0.15);
}

.glow-indigo {
  box-shadow: 0 0 35px -5px rgba(99, 102, 241, 0.25);
}

.glow-emerald {
  box-shadow: 0 0 35px -5px rgba(16, 185, 129, 0.25);
}

.glow-rose {
  box-shadow: 0 0 35px -5px rgba(244, 63, 94, 0.25);
}

/* Float Animation */
@keyframes float {
  0% { transform: translateY(0px) rotate(0deg); }
  50% { transform: translateY(-12px) rotate(0.5deg); }
  100% { transform: translateY(0px) rotate(0deg); }
}

.animate-float {
  animation: float 6s ease-in-out infinite;
}

/* Pulse Glow Animation */
@keyframes pulse-glow {
  0% { opacity: 0.3; transform: scale(0.98); }
  50% { opacity: 0.6; transform: scale(1.02); }
  100% { opacity: 0.3; transform: scale(0.98); }
}

.animate-pulse-glow {
  animation: pulse-glow 8s ease-in-out infinite;
}

/* Scrollbar styles */
::-webkit-scrollbar {
  width: 8px;
  height: 8px;
}
::-webkit-scrollbar-track {
  background: rgba(10, 10, 15, 0.5);
}
::-webkit-scrollbar-thumb {
  background: rgba(255, 255, 255, 0.1);
  border-radius: 4px;
}
::-webkit-scrollbar-thumb:hover {
  background: rgba(99, 102, 241, 0.3);
}
