"""Embed a web chart (bs3/web/charts.py) as an <iframe srcdoc>: plot_html builds both theme variants and embeds them in
one iframe; a small script picks the variant that matches the Gradio theme, waits until the iframe is actually visible
before drawing (charts in hidden tabs must not be measured at width 0), and re-draws when the theme changes. The chart
uses the page font (Source Sans Pro of the Gradio Default theme): the iframe loads it itself and draws once it is there
(plotly measures legend and label widths with the font it has)."""
from __future__ import annotations

import html as _html
import json

from ..palette import FONT_FAMILY, THEME

# the page font for the chart iframes (the same Google Fonts file as the Gradio theme, so it comes from the cache)
FONT_CSS = "https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600&display=swap"


# ---------------------------------------------------------------- iframe embedding
_FRAME_JS = r"""
(function(){
var SPEC=__SPEC__, EXTRA=__EXTRA__, FILL=__FILL__, FONT=__FONT__, INK=__INK__;
var gd=document.getElementById('g'), cur=null, curK=0, fitTimer=null, fitRuns=0, lastW=-1, fontReady=false;
var lastH=-1, natH=null, maxH=null;         // FILL only: last viewport height, natural and largest figure height
var CFG={responsive:true, displaylogo:false, locale:'ru',
  modeBarButtonsToRemove:['select2d','lasso2d','autoScale2d','zoomIn2d','zoomOut2d','toggleSpikelines',
    'hoverClosestCartesian','hoverCompareCartesian','toImage']};
// plotly's own texts (toolbar tooltips, the zoom hint) in Russian; no number format here, so 0.25 stays 0.25
if(typeof Plotly!=='undefined'){ try{ Plotly.register({moduleType:'locale', name:'ru', dictionary:{
  'Zoom':'Увеличить область', 'Pan':'Сдвигать', 'Reset axes':'Вернуть исходный масштаб', 'Reset views':'Вернуть исходный вид',
  'Reset view':'Вернуть исходный вид', 'Reset':'Сбросить', 'Autoscale':'Масштаб по данным', 'Zoom in':'Приблизить',
  'Zoom out':'Отдалить', 'Box Select':'Выделить прямоугольником', 'Lasso Select':'Выделить лассо',
  'Toggle Spike Lines':'Линии к осям', 'Show closest data on hover':'Подсказка по ближайшей точке',
  'Compare data on hover':'Подсказка по всем линиям', 'Toggle show closest data on hover':'Подсказка по ближайшей точке',
  'Download plot as a PNG':'Сохранить как картинку', 'Download plot':'Сохранить график',
  'Double-click to zoom back out':'Двойной щелчок — вернуть исходный масштаб',
  'Double-click on legend to isolate one trace':'Двойной щелчок по легенде — оставить только эту линию',
  'Taking snapshot - this may take a few seconds':'Сохраняю картинку, это может занять несколько секунд',
  'Snapshot succeeded':'Картинка сохранена', 'Sorry, there was a problem downloading your image!':'Не удалось сохранить картинку',
  'Produced with Plotly.js':'Построено библиотекой графиков', 'Share chart...':'Поделиться графиком'}}); }catch(e){} }
function isDark(){
  var readable=false;
  try{ for(var n=window.frameElement; n; n=n.parentElement){ readable=true; if(n.classList && n.classList.contains('dark')) return true; } }catch(e){}
  try{ var d=window.parent.document; if(d && d!==document){ readable=true;
    if((d.body && d.body.classList.contains('dark')) || d.documentElement.classList.contains('dark')) return true; } }catch(e){}
  if(readable) return false;
  try{ return window.matchMedia('(prefers-color-scheme: dark)').matches; }catch(e){ return true; }
}
function copy(o){ return JSON.parse(JSON.stringify(o)); }
// Visible = the iframe element itself has a box in the parent page. The iframe's own clientWidth is not enough: once a
// Gradio tab has been shown and hidden again, the hidden iframe keeps its old non-zero width.
function visible(){
  try{ var fe=window.frameElement;
    if(fe && (fe.offsetWidth<=0 || fe.getClientRects().length===0)) return false; }catch(e){}
  return document.documentElement.clientWidth>0;
}
var pollTimer=null;
function waitVisible(){
  if(pollTimer) return;
  pollTimer=setInterval(function(){ if(visible()){ clearInterval(pollTimer); pollTimer=null; draw(); } }, 300);
}
// FILL: the iframe height is the chart's natural height (its flex basis), never the drawn one, so a window stretched
// to its neighbour's height can shrink back when the neighbour gets shorter. maxH stops the growth (radar), the page
// CSS then centres the chart with its subtitle in the window.
function sizeFrame(){
  try{ var fe=window.frameElement, fl=gd._fullLayout; if(!fe || !fl) return;
    var h=Math.ceil((FILL && natH ? natH : fl.height)+EXTRA)+'px', mh=FILL && maxH ? Math.ceil(maxH+EXTRA)+'px' : '';
    if(fe.style.height===h && fe.style.maxHeight===mh) return;
    fe.style.height=h; fe.style.maxHeight=mh;
    // FILL: the figure follows the new iframe height; a hidden browser tab fires no resize event until it is shown
    if(FILL) scheduleFit(); }catch(e){}
}
function scheduleFit(){ clearTimeout(fitTimer); fitTimer=setTimeout(fit, 220); }
// A legend entry wider than the chart is cut off (plotly never wraps it): break such names before their "(…)" part.
function legendWrap(W){
  var over=false;
  gd.querySelectorAll('.legend .legendtext').forEach(function(t){ if(t.getBoundingClientRect().right>W-1) over=true; });
  if(!over) return null;
  var names=[], idx=[];
  (gd.data||[]).forEach(function(tr, i){ var n=tr.name||'';
    if(tr.showlegend!==false && n.indexOf('<br>')<0 && n.indexOf(' (')>0){ names.push(n.replace(' (', '<br>(')); idx.push(i); } });
  return idx.length ? {names:names, idx:idx} : null;
}
function fit(){
  var fl=gd._fullLayout, L=gd.layout; if(!fl || !fl._size || !L) return;
  if(!visible()) return;
  // Plotly pins layout.width as soon as the height has been relaid and then ignores container resizes
  // (Plots.resize does nothing when both are set), so the width follows the iframe here.
  var W=document.documentElement.clientWidth;
  if(W>0 && Math.abs(fl.width-W)>1){ Plotly.relayout(gd, {width:W}).then(function(){ fit(); }); return; }
  var wrap=legendWrap(W);
  if(wrap){ Plotly.restyle(gd, {name:wrap.names}, wrap.idx).then(function(){ fit(); }); return; }
  var m=L.meta||{}, upd={}, tt=m.time_ticks;
  if(L.polar && L.polar.radialaxis){
    // radar: the scale labels sit 0.2·r apart; on a small circle only every other label is kept
    var r=Math.min(fl._size.w, 0.9*fl._size.h)/2;
    var pt=r*0.2>=24 ? ['0.2','0.4','0.6','0.8','1'] : ['0.2','','0.6','','1'];
    if(pt.join()!==(L.polar.radialaxis.ticktext||[]).join()) upd['polar.radialaxis.ticktext']=pt;
  }
  if(tt && tt.vals && tt.vals.length>1){
    var need=tt.min_px*tt.vals.length/Math.max(40, fl._size.w), k=tt.steps[tt.steps.length-1];
    for(var i=0;i<tt.steps.length;i++){ if(tt.steps[i]>=need){ k=tt.steps[i]; break; } }
    if(k!==curK){ var v=[], t=[]; for(var j=0;j<tt.vals.length;j+=k){ v.push(tt.vals[j]); t.push(tt.text[j]); }
      tt.axes.forEach(function(a){ upd[a+'.tickvals']=v; upd[a+'.ticktext']=t; }); curK=k; }
  }
  if(m.plot_h && fitRuns<6){
    // radar in a narrow column: the circle is limited by the width, so do not keep empty space above and below it
    var target=L.polar ? Math.max(200, Math.min(m.plot_h, Math.round(fl._size.w)+60)) : m.plot_h;
    var diff=target-fl._size.h;
    if(FILL){
      // natural height as above; in a window stretched by its row the figure grows with the iframe. The radar grows
      // only while its circle can grow (plot height up to the width + 60, as above), so no empty band opens between
      // the legend and the circle
      natH=Math.round(Math.max(160, fl.height+diff));
      maxH=L.polar ? Math.max(natH, Math.round(fl.height-fl._size.h+fl._size.w+60)) : null;
      var want=Math.max(natH, Math.min(window.innerHeight-EXTRA, maxH || Infinity));
      if(Math.abs(want-fl.height)>2){ upd.height=want; fitRuns++; }
    } else if(Math.abs(diff)>2){ upd.height=Math.round(Math.max(160, fl.height+diff)); fitRuns++; } }
  if(Object.keys(upd).length){ Plotly.relayout(gd, upd).then(function(){ sizeFrame(); if('height' in upd) scheduleFit(); }); }
  else sizeFrame();
}
function draw(){
  if(!fontReady) return;                // the first draw comes from the font loader at the end of this script
  if(typeof Plotly==='undefined'){
    gd.textContent='График не загрузился: библиотека графиков загружается из интернета, а доступа к нему нет';
    gd.style.cssText='font:14px '+FONT+';padding:8px;color:'+INK[isDark()?'dark':'light']; return; }
  var mode=isDark()?'dark':'light';
  // hidden tab: plotly would measure text as 0x0 (overlapping legend items, lost bar labels), so wait until shown
  if(!visible()){ if(mode!==cur) waitVisible(); return; }
  var w=document.documentElement.clientWidth, h=window.innerHeight;
  if(w!==lastW || (FILL && h!==lastH)){ lastW=w; lastH=h; fitRuns=0; }
  if(mode===cur){ scheduleFit(); return; }
  cur=mode; curK=0; fitRuns=0;
  document.documentElement.setAttribute('data-theme', mode);
  var F=SPEC[mode];
  Plotly.newPlot(gd, copy(F.data), copy(F.layout), CFG).then(function(){ fit(); });
}
// unified hover: show the segment ("отрезок 5:00–5:20") instead of the raw midpoint in seconds
new MutationObserver(function(){
  var L=gd.layout; if(!L || !L.meta || !L.meta.seg_hover) return;
  var t=gd.querySelector('.hoverlayer .legendtitletext'); if(!t) return;
  var hd=gd._hoverdata||[], lab=null;
  for(var i=0;i<hd.length && lab===null;i++){ var cd=hd[i].customdata; if(Array.isArray(cd)) cd=cd[0]; if(typeof cd==='string') lab=cd; }
  if(lab===null) return;
  var txt='отрезок '+lab; if(t.textContent===txt) return;
  t.textContent=txt;
  try{ var box=t.closest('.legend'), bg=box && box.querySelector('rect.bg');
    if(bg){ var over=t.getBoundingClientRect().right+8-bg.getBoundingClientRect().right;
      if(over>0) bg.setAttribute('width', parseFloat(bg.getAttribute('width'))+over); } }catch(e){}
}).observe(gd, {childList:true, subtree:true, characterData:true});
try{ new ResizeObserver(draw).observe(document.documentElement); }catch(e){ window.addEventListener('resize', draw); }
// the observer sees only width changes (the document is as tall as the chart); stretching the iframe changes its height
if(FILL) window.addEventListener('resize', draw);
try{ var pd=window.parent.document; if(pd && pd!==document){
  var mo=new MutationObserver(draw);
  mo.observe(pd.body, {attributes:true, attributeFilter:['class']});
  mo.observe(pd.documentElement, {attributes:true, attributeFilter:['class']}); } }catch(e){}
try{ var mq=window.matchMedia('(prefers-color-scheme: dark)');
  if(mq.addEventListener) mq.addEventListener('change', draw); else mq.addListener(draw); }catch(e){}
// draw once the page font is loaded (Cyrillic and Latin faces, regular and semibold), but never wait long: without a
// connection plotly draws with the fallback font
var ready=Promise.resolve();
try{ if(document.fonts && document.fonts.load){
  var loads=['400 13px "Source Sans Pro"','600 13px "Source Sans Pro"'].map(function(f){ return document.fonts.load(f, 'Жж Aa 0'); });
  ready=Promise.race([Promise.all(loads), new Promise(function(res){ setTimeout(res, 1500); })]); } }catch(e){}
function start(){ fontReady=true; draw(); }
ready.then(start, start);
})();
"""


def plot_html(builder, rep: dict | None = None, extra_height: int = 24, fill: bool = False) -> str:
    """builder(rep, theme) -> figure. Both theme variants go into one <iframe srcdoc>; the chart title is rendered as
    HTML above the iframe (gr.HTML shows no label), so it uses the page text colour of the current Gradio theme.
    fill=True: for a chart in one of two windows side by side. The iframe may grow (flex, see style.APP_CSS) to the
    height of the neighbouring window and the figure grows with it instead of leaving an empty band under the chart;
    a radar grows only while its circle can grow, then the page centres it."""
    import plotly.io as pio
    from plotly.offline import get_plotlyjs_version

    if callable(builder):
        figs = {"dark": builder(rep, "dark"), "light": builder(rep, "light")}
    else:                                            # a ready figure: the same spec for both themes
        figs = {"dark": builder, "light": builder}
    lt = figs["dark"].layout.title
    title = (lt.text or "").strip()
    subtitle = ((lt.subtitle.text if lt.subtitle else "") or "").strip()
    height = int(figs["dark"].layout.height or 360)
    specs = []
    for name in ("dark", "light"):
        fig = figs[name]
        fig.update_layout(title=None)
        specs.append(f'"{name}":' + pio.to_json(fig, validate=False, remove_uids=True))
    spec = ("{" + ",".join(specs) + "}").replace("</", "<\\/").replace("<!--", "<\\!--")
    js = (_FRAME_JS.replace("__SPEC__", spec).replace("__EXTRA__", str(int(extra_height)))
          .replace("__FILL__", "true" if fill else "false").replace("__FONT__", json.dumps(FONT_FAMILY))
          .replace("__INK__", json.dumps({t: THEME[t]["text"] for t in ("dark", "light")})))
    cdn = f"https://cdn.plot.ly/plotly-{get_plotlyjs_version()}.min.js"
    # radar scale labels (0.2 … 1) sit on top of the data lines: a background-coloured outline keeps them legible
    halo = ".radial-axis text{paint-order:stroke;stroke-width:3px;stroke-linejoin:round}" + "".join(
        f"html[data-theme={t}] .radial-axis text{{stroke:{THEME[t]['sep']}}}" for t in ("dark", "light"))
    doc = ("<!doctype html><html><head><meta charset='utf-8'>"
           f"<link rel='stylesheet' href='{FONT_CSS}'>"
           f"<style>html,body{{margin:0;padding:0;background:transparent;overflow:hidden}}#g{{width:100%}}{halo}</style>"
           f"<script src='{cdn}' charset='utf-8'></script></head><body><div id='g'></div><script>{js}</script>"
           "</body></html>")
    srcdoc = doc.replace("&", "&amp;").replace('"', "&quot;")
    head = ""
    if title:
        head = f"<div style='font-size:15px;font-weight:600;line-height:1.35;margin:10px 0 2px'>{_html.escape(title)}</div>"
        if subtitle:
            head += f"<div style='font-size:13px;line-height:1.35;opacity:.85;margin:0 0 4px'>{_html.escape(subtitle)}</div>"
    grow = "flex:1 0 auto;" if fill else ""
    return (head + f"<iframe style='width:100%;height:{height + int(extra_height)}px;{grow}border:0;display:block' "
            f"scrolling='no' srcdoc=\"{srcdoc}\"></iframe>")
