INDEX_HTML = """<!doctype html>
<html><head><meta charset="utf-8"><title>Framework Agent</title>
<style>
 body{font-family:system-ui,sans-serif;max-width:640px;margin:40px auto;padding:0 16px}
 #log{border:1px solid #ddd;border-radius:8px;padding:12px;min-height:240px;margin-bottom:12px}
 .msg{margin:8px 0}.you{color:#1a5}.bot{color:#222;white-space:pre-wrap}.meta{color:#999;font-size:.8rem}
 form{display:flex;gap:8px}input{flex:1;padding:8px;border:1px solid #ccc;border-radius:6px}
 button{padding:8px 16px;border:0;border-radius:6px;background:#1a5;color:#fff;cursor:pointer}
</style></head><body>
<h1>Framework Agent</h1><div id="log"></div>
<form id="f"><input id="m" placeholder="Ask something..." autofocus><button>Send</button></form>
<script>
const log=document.getElementById('log'),form=document.getElementById('f'),input=document.getElementById('m');
const tid='web-'+Math.random().toString(36).slice(2,8);
function add(c,t){const d=document.createElement('div');d.className='msg '+c;d.textContent=t;log.appendChild(d);log.scrollTop=log.scrollHeight;return d;}
form.onsubmit=async e=>{
 e.preventDefault();const msg=input.value.trim();if(!msg)return;
 add('you','You: '+msg);input.value='';
 const bot=add('bot','Nova: ');
 const res=await fetch('/chat/stream?thread_id='+tid,{method:'POST',headers:{'content-type':'application/json'},body:JSON.stringify({message:msg})});
 const reader=res.body.getReader(),dec=new TextDecoder();
 while(true){const {value,done}=await reader.read();if(done)break;
  dec.decode(value).split('\\n\\n').forEach(line=>{if(line.startsWith('data: '))bot.textContent+=line.slice(6);});}
};
</script></body></html>"""
