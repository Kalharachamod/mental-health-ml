from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
import os

def build_report(pdf_path, inputs:dict, outputs:dict, tips:list, anomaly_note:str=""):
    os.makedirs(os.path.dirname(pdf_path),exist_ok=True)
    c=canvas.Canvas(pdf_path,pagesize=A4)
    w,h=A4; y=h-50

    c.setFont("Helvetica-Bold",14)
    c.drawString(50,y,"Mental Health Risk Report"); y-=30
    c.setFont("Helvetica",10)

    c.drawString(50,y,"Inputs:"); y-=15
    for k,v in inputs.items():
        c.drawString(60,y,f"- {k}: {v}"); y-=12

    y-=10; c.drawString(50,y,"Outputs:"); y-=15
    if "Condition Index" in outputs:
        c.drawString(60,y,f"- Mental Health Index: {outputs['Condition Index']}/100"); y-=12
        c.drawString(60,y,f"- Risk Category: {outputs['Risk Category']}"); y-=12
        c.drawString(60,y,f"- Confidence Score: {outputs['Condition Risk']}"); y-=12
    if "Stress Level" in outputs:
        c.drawString(60,y,f"- Stress Level: {outputs['Stress Level']}"); y-=12
    if "Severity" in outputs:
        c.drawString(60,y,f"- Severity: {outputs['Severity']}"); y-=12

    if anomaly_note:
        y-=10; c.drawString(50,y,"Anomaly:"); y-=15
        c.drawString(60,y,anomaly_note); y-=12

    y-=10; c.drawString(50,y,"Recommendations:"); y-=15
    for t in tips:
        c.drawString(60,y,f"- {t}"); y-=12

    c.showPage(); c.save()
