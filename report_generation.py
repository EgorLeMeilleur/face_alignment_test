from fpdf import FPDF
from pathlib import Path
import config

class PDFReport(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Face Alignment Experiments Report', ln=True, align='C')

    def footer(self):
        self.set_y(-15)
        self.set_font('Arial', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_report():
    pdf = PDFReport()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    pdf.cell(0, 10, "Experiments Summary:", ln=True)
    for exp in config.EXPERIMENTS:
        pdf.cell(0, 10, f"Experiment: {exp['name']} - Model: {exp['model_type']}, Loss: {exp['loss_type']}", ln=True)
        auc_log_file = Path(config.RESULTS_DIR) / f"auc_results_{exp['name']}.txt"
        if auc_log_file.exists():
            with open(auc_log_file, "r") as f:
                auc_lines = f.readlines()
            for line in auc_lines:
                pdf.cell(0, 10, line.strip(), ln=True)
    else:
        pdf.cell(0, 10, "No AUC results found.", ln=True)
    
    pdf.ln(10)
    pdf.cell(0, 10, "CED graphs for 300W and Menpo datasets are attached below.", ln=True)
    
    for ds in ["300W", "Menpo"]:
        for exp in config.EXPERIMENTS:
            img_path = Path(config.RESULTS_DIR) / f"CED_{ds}_{exp['name']}.png"
            if img_path.exists():
                pdf.add_page()
                pdf.cell(0, 10, f"CED Curve for {ds} dataset - {exp['name']}", ln=True)
                pdf.image(str(img_path), x=10, y=20, w=pdf.w - 20)
    
    report_file = Path(config.REPORT_DIR)
    report_file.mkdir(parents=True, exist_ok=True)
    output_pdf = report_file / "experiment_report.pdf"
    pdf.output(str(output_pdf))
    print(f"Report saved to {output_pdf}")

if __name__ == "__main__":
    generate_report()
