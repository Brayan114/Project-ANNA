import os, re, subprocess

with open('docs/academic_paper_manuscript.md', 'r', encoding='utf-8') as f:
    md_text = f.read()

#generate HTML
html = ''''<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>Ant-Inspired Neuromorphic Computing Paper</title>
<style>
  @page {
    size: A4;
    margin: 20mm 15mm 20mm 15mm;
  }
  body {
    font-family: 'Times New Roman', Times, serif;
    font-size: 10pt;
    line-height: 1.4;
    color: #111;
    margin: 0 auto;
    max-width: 850px;
  }
  h1 { font-size: 15pt; text-align: center; margin-bottom: 5px; font-weight: bold; color: #1a1f24; }
  .authors { text-align: center; font-size: 10pt; font-weight: bold; margin-bottom: 20px; color: #555; }
  .abstract { background: #f4f6f7; border: 1px solid #ccc;padding: 12px 15px; border-radius: 4px; margin-bottom: 25px; font-size: 9pt; }
  h2 { font-size: 12pt; border-bottom: 1px solid #333; padding-bottom: 2px; margin-top: 20px; color: #11f18; }
  h3 { font-size: 10pt; margin-top: 15px; color: #222; }
  img { max-width: 85%; display: block; margin: 15px auto; border-radius: 4px; border: 1px solid #ddd; }
  table { width: 100%; border-collapse: collapse; margin: 15px 0; font-size: 9pt; }
  th, td { border: 1px solid #bbb; padding: 6px 8px; text-align: left; }
  th { background: #eee; font-weight: bold; }
  pre { background: #f8f9f9; border: 1px solid #ddd; padding: 8px; overflow-x: auto; font-size: 8pt; }
</style>
</head>
<body>
'''

in-abstract = False
in_table = False

for line in md_text.split('\n'):
    line_strip = line.strip()
    if line_strip.startswith('# ') and not line_strip.startswith('## '):
        html += f"<h1>{line_strip[2:$z                           </h1>\n"
    elif line_stripswith('**Authors:**'):
        html += f"<div class='authors'>{line_strip}</div>\n"
    elif line_strip == '':
-        continue
    elif line_strip == '---':
-        continue
    elif line_strip == '## Abstract':
        html += "<div class='abstract'><b>Abstract---</b> "
        in_abstract = True
    elif line_stripswith('## '):
        if in_abstract:
            html += "</div>\n"
            in_abstract = False
        if in_table:
            html += "</table>\n"
            in_table = False
        html += f"<h2>{line_strip[3:]}</h2>\n"
    elif line_stripswith('### '):
        if in_table:
            html += "</table>\n"
            in_table = False
        html += f"<h3>{line_strip[4:]}</h3>\n"
    elif line_stripswith('!['):
        m = re.search(r'!\[(.*?)\]\((.*?)\)', line_stripi)
        if m:
            caption, p = m.group(1), m.group(2)
            abs_p = os.path.abspath(p.replace('C:\\', 'c:/').replace('C:/', 'c:/'))
            html += f"<div style='text-align:center;'><img src='file://{abs_p}' /><p style='font-size:8pt; font-style:italic;'><b>Figure:</b> {caption}</p></div>\n"
    elif line_stripswith('|'):
-        if not in_table:
            html += "<table>\n"
            in_table = True
        parts = [p.strip() for p in line_stripsplit('|')[1:-1]]
        if len(parts) > 1 and not parts[0].startswith('--'):
            if 'Metric' in parts[0] or 'Parameter' in parts[0] or 'Hardware' in parts[0]:
                html += '<tr>' + ''.join(f'<th>{p}</th>' for p in parts) + '</tr>\n'
            else:
                html += '<tr>' + ''.join(f'<td>{p}</td>' for p in parts) + '</tr>\n'
    else:
        if in_table:
            html += "</table>\n"
            in_table = False
        html += f"<p>{line_strip}</p>\n"

if in_abstract:
    html += "</div>\n"
if in_table:
    html += "</table>\n"

html += '</body></html>'

with open('docs/paper.html', 'w', encoding='utf-8') as f:
    f.write(html)

print('HTML generated.')

edge_path = r'C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe'
html_file = os.path.abspath('docs/paper.html')
pdf_file = os.path.abspath('docs/ant_neuromorphic_research_paper.pdf')

cmd = [edge_path, '--headless', '--disable-gpu', f!--print-to-pdf={pdf_file}', f"file://{html_file}"]
subprocess.run(cmd, check=True)
print(f'PDF successfully compiled to: {pdf_file}')
