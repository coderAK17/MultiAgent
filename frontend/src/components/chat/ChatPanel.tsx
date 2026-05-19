import { useEffect, useRef, useState } from "react";
import { Send, Paperclip, Sparkles, Mic } from "lucide-react";
import ReactMarkdown from "react-markdown";
import { motion, AnimatePresence } from "framer-motion";
import { useAppStore } from "@/store/useAppStore";
import { cn } from "@/lib/utils";
import type { ChatMessage, AgentId } from "@/types";
import { chat } from "@/lib/api";

export function ChatPanel() {
  const messages = useAppStore((s) => s.messages);
  const pushMessage = useAppStore((s) => s.pushMessage);
  const pushActivity = useAppStore((s) => s.pushActivity);
  const setAgentStatus = useAppStore((s) => s.setAgentStatus);
  const addDocument = useAppStore((s) => s.addDocument);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const endRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: "smooth" }); }, [messages]);

  const runChat = async (userText: string) => {
    setBusy(true);
    
    try {
      setAgentStatus("classifier", "working", "Analyzing request");
      pushActivity({ id: crypto.randomUUID(), agentId: "classifier", title: "Analyzing request", at: Date.now(), level: "info" });
      
      const response = await chat(userText);
      
      setAgentStatus("classifier", "done", undefined);
      
      const reply: ChatMessage = {
        id: crypto.randomUUID(),
        role: "assistant",
        agentId: response.agents?.[0] as AgentId | undefined,
        content: response.reply,
        createdAt: Date.now(),
      };
      pushMessage(reply);
    } catch (e) {
      pushMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        content: "Sorry, I ran into an error connecting to the backend.",
        createdAt: Date.now()
      });
    } finally {
      setBusy(false);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file || busy) return;
    
    setBusy(true);
    try {
      setAgentStatus("classifier", "working", "Analyzing document");
      pushActivity({ id: crypto.randomUUID(), agentId: "classifier", title: "Analyzing document", at: Date.now(), level: "info" });
      
      const { uploadDocument } = await import("@/lib/api");
      const response = await uploadDocument(file);
      
      const docRecord: import("@/types").DocumentRecord = {
        id: crypto.randomUUID(),
        name: file.name,
        category: response.classification.category,
        uploadedAt: Date.now(),
        status: "ready",
        summary: response.medical?.summary ?? response.research?.summary,
        analysis: response
      };
      addDocument(docRecord);
      
      setAgentStatus("classifier", "done", undefined);
      
      pushMessage({
        id: crypto.randomUUID(),
        role: "user",
        content: `Uploaded document: **${file.name}**`,
        createdAt: Date.now(),
      });
      
      pushMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        agentId: "classifier",
        content: `I've analyzed **${file.name}**.\nIt was classified as **${response.classification.category}**.\n\nSummary:\n${docRecord.summary ?? "Processed successfully."}`,
        createdAt: Date.now(),
      });
      
    } catch (error) {
      pushMessage({
        id: crypto.randomUUID(),
        role: "assistant",
        content: "Sorry, there was an error uploading the document.",
        createdAt: Date.now()
      });
    } finally {
      setBusy(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  };

  const send = async () => {
    if (!input.trim() || busy) return;
    const userMsg: ChatMessage = { id: crypto.randomUUID(), role: "user", content: input, createdAt: Date.now() };
    pushMessage(userMsg);
    const text = input;
    setInput("");
    await runChat(text);
  };

  return (
    <div className="flex h-full flex-col">
      <div className="flex items-center justify-between border-b border-border/60 px-6 py-4">
        <div>
          <h1 className="font-display text-lg font-semibold">Intelligence workspace</h1>
          <p className="text-xs text-muted-foreground">Multi-agent · streaming · context-preserving</p>
        </div>
        <span className="glass inline-flex items-center gap-1.5 rounded-full px-3 py-1 text-xs">
          <Sparkles className="h-3 w-3 text-accent" /> Live
        </span>
      </div>

      <div className="flex-1 overflow-y-auto px-6 py-8">
        {messages.length === 0 ? <EmptyState /> : (
          <div className="mx-auto max-w-3xl space-y-6">
            <AnimatePresence initial={false}>
              {messages.map((m) => <MessageBubble key={m.id} m={m} />)}
            </AnimatePresence>
            {busy && <TypingIndicator />}
            <div ref={endRef} />
          </div>
        )}
      </div>

      <div className="border-t border-border/60 p-4">
        <div className="mx-auto max-w-3xl">
          <div className="glass-strong flex items-end gap-2 rounded-2xl p-2">
            <input type="file" ref={fileInputRef} className="hidden" onChange={handleFileUpload} />
            <button onClick={() => fileInputRef.current?.click()} className="rounded-lg p-2 text-muted-foreground hover:bg-secondary hover:text-foreground"><Paperclip className="h-4 w-4" /></button>
            <textarea
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => { if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); } }}
              placeholder="Ask the agents anything, or drop a document…"
              rows={1}
              className="max-h-40 flex-1 resize-none bg-transparent px-2 py-2 text-sm outline-none placeholder:text-muted-foreground"
            />
            <button className="rounded-lg p-2 text-muted-foreground hover:bg-secondary hover:text-foreground"><Mic className="h-4 w-4" /></button>
            <button onClick={send} disabled={!input.trim() || busy} className="rounded-lg p-2 text-primary-foreground transition-opacity disabled:opacity-40" style={{ background: "var(--gradient-hero)" }}>
              <Send className="h-4 w-4" />
            </button>
          </div>
          <p className="mt-2 text-center text-[10px] text-muted-foreground">Press <kbd className="rounded bg-secondary px-1">Enter</kbd> to send · <kbd className="rounded bg-secondary px-1">Shift+Enter</kbd> for newline</p>
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  const prompts = [
    "Draft a plan for a new marketing campaign",
    "Simulate a CEO and CTO discussing our next feature",
    "Research the latest trends in autonomous agents",
    "Extract abnormal values from my CBC report",
  ];
  return (
    <div className="mx-auto max-w-2xl text-center">
      <motion.div initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }} className="mx-auto mb-6 flex h-14 w-14 items-center justify-center rounded-2xl" style={{ background: "var(--gradient-hero)" }}>
        <Sparkles className="h-6 w-6 text-background" />
      </motion.div>
      <h2 className="font-display text-2xl font-semibold tracking-tight">How can the agents help?</h2>
      <p className="mt-2 text-sm text-muted-foreground">Upload a document or start with one of these prompts.</p>
      <div className="mt-6 grid gap-2 sm:grid-cols-2">
        {prompts.map((p) => (
          <button key={p} className="glass rounded-xl px-4 py-3 text-left text-sm text-muted-foreground hover:text-foreground transition-colors">
            {p}
          </button>
        ))}
      </div>
    </div>
  );
}

function MessageBubble({ m }: { m: ChatMessage }) {
  const isUser = m.role === "user";
  return (
    <motion.div
      initial={{ opacity: 0, y: 8 }}
      animate={{ opacity: 1, y: 0 }}
      className={cn("flex gap-3", isUser && "flex-row-reverse")}
    >
      <div
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg text-[10px] font-semibold uppercase",
          isUser ? "bg-secondary text-foreground" : "text-background",
        )}
        style={!isUser ? { background: "var(--gradient-hero)" } : undefined}
      >
        {isUser ? "U" : "AI"}
      </div>
      <div className={cn("min-w-0 max-w-[85%] rounded-2xl px-4 py-3 text-sm", isUser ? "bg-primary/15 border border-primary/20" : "glass")}>
        {m.agentId && !isUser && (
          <p className="mb-1 text-[10px] uppercase tracking-wider text-accent">{m.agentId} agent</p>
        )}
        <div className="prose prose-sm prose-invert max-w-none prose-p:my-1 prose-strong:text-foreground prose-li:my-0">
          <ReactMarkdown>{m.content}</ReactMarkdown>
        </div>
      </div>
    </motion.div>
  );
}

function TypingIndicator() {
  return (
    <div className="flex items-center gap-3">
      <div className="flex h-8 w-8 items-center justify-center rounded-lg text-background" style={{ background: "var(--gradient-hero)" }}>
        <Sparkles className="h-4 w-4" />
      </div>
      <div className="glass rounded-2xl px-4 py-3">
        <div className="flex items-center gap-1">
          {[0,1,2].map(i => (
            <motion.span key={i} className="h-1.5 w-1.5 rounded-full bg-primary"
              animate={{ opacity: [0.3, 1, 0.3] }}
              transition={{ duration: 1, repeat: Infinity, delay: i * 0.15 }}
            />
          ))}
        </div>
      </div>
    </div>
  );
}
