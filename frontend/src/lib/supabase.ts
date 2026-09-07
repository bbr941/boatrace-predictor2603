import { createClient } from '@supabase/supabase-js';

const supabaseUrl = import.meta.env.VITE_SUPABASE_URL || 'https://xbarsfkzfmuaamsiksdl.supabase.co';
const supabaseAnonKey = import.meta.env.VITE_SUPABASE_ANON_KEY || '';

export const isSupabaseConfigured = Boolean(
  supabaseUrl && 
  supabaseAnonKey && 
  supabaseAnonKey !== 'your_supabase_anon_key_here' &&
  supabaseAnonKey !== 'YOUR_SUPABASE_ANON_KEY_HERE'
);

// Fallback dummy client if key is not configured to avoid initialization crash
export const supabase = createClient(
  supabaseUrl,
  supabaseAnonKey || 'dummy_anon_key_for_initialization'
);

export const getSupabaseConfigInfo = () => ({
  url: supabaseUrl,
  hasKey: isSupabaseConfigured,
});
