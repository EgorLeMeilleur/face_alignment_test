from fpdf import FPDF
from pathlib import Path
import config

class PDFReport(FPDF):
    def header(self):
        self.set_font('DejaVu', 'B', 12)
        self.cell(0, 10, 'Face Alignment Experiments Report', ln=True, align='C')

    def footer(self):
        self.set_y(-15)
        self.set_font('DejaVu', 'I', 8)
        self.cell(0, 10, f'Page {self.page_no()}', align='C')

def generate_report():
    pdf = PDFReport()

    pdf.add_font('DejaVu', '', 'DejaVuSans.ttf', uni=True)
    pdf.add_font('DejaVu', 'B', 'DejaVuSans-Bold.ttf', uni=True)
    pdf.add_font('DejaVu', 'I', 'DejaVuSans-Oblique.ttf', uni=True)
    
    pdf.add_page()
    pdf.set_font("DejaVu", size=12)

    
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
    
    pdf.add_page()
    pdf.ln(10)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 10, "Описание экспериментов", ln=True)
    pdf.set_font("DejaVu", size=12)
    description_text = (
        "В проведенных экспериментах сравнивались различные архитектуры моделей с предобученными весами "
        "efficientvit, efficientnet и convnext. Для каждой архитектуры тестировались два типа функций потерь - MSE и Wing. "
        "Каждая модель обучалась с использованием одних и тех же настроек обработки данных и аугментаций, что позволило "
        "объективно сравнить их эффективность по метрике AUC на наборах данных 300W и Menpo."
    )
    pdf.multi_cell(0, 10, description_text)
    
    pdf.ln(10)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 10, "Выводы", ln=True)
    pdf.set_font("DejaVu", size=12)
    conclusion_text = (
        "На основании полученных результатов можно сделать следующие выводы:\n"
        "1. Модели, использующие функцию потерь Wing, как правило, демонстрируют более высокие значения AUC, "
        "особенно на наборе данных Menpo.\n"
        "2. Среди протестированных архитектур, efficientnet с функцией потерь Wing показывает наилучшие результаты.\n"
        "3. Применение продвинутых методов аугментации и корректной обработки данных способствует повышению "
        "стабильности и точности модели."
    )
    pdf.multi_cell(0, 10, conclusion_text)

    pdf.ln(10)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 10, "Гиперпараметры", ln=True)
    pdf.set_font("DejaVu", size=12)
    hyperparams_text = (
        f"BATCH_SIZE = {config.BATCH_SIZE}\n"
        f"NUM_WORKERS = {config.NUM_WORKERS}\n"
        f"LEARNING_RATE = {config.LEARNING_RATE}\n"
        f"EPOCHS = {config.EPOCHS}\n"
        f"IMAGE_SIZE = {config.IMAGE_SIZE}\n"
        f"NUM_POINTS = {config.NUM_POINTS}\n"
        f"TRAIN_VAL_SPLIT = {config.TRAIN_VAL_SPLIT}\n"
        f"CROP_EXPANSION = {config.CROP_EXPANSION}\n"
        f"MAX_ERROR_THRESHOLD = {config.MAX_ERROR_THRESHOLD}\n"
        f"LOSS_TYPE = {config.LOSS_TYPE}\n"
        f"MODEL_TYPE = {config.MODEL_TYPE}"
    )
    pdf.multi_cell(0, 10, hyperparams_text)
    
    pdf.ln(10)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 10, "Обработка данных", ln=True)
    pdf.set_font("DejaVu", size=12)
    data_processing_text = (
        "В качестве прямоугольника лица брались модифицированные прямоугольники dlib: "
        "левый верхний угол - минимум из левого верхнего угла прямоугольника dlib и самой левой и верхей ключевой точки, "
        "аналогично для правого нижнего угла. Также прямоугольник немного расширялся на заданный в конфиге параметр. "
        "В качестве аугментаций были использованы RandomBrightnessContrast, GaussianBlur и ToGray. "
        "Нормализация происходила со значениями датасета ImageNet, так как использовались предобученные модели."
    )
    pdf.multi_cell(0, 10, data_processing_text)

    pdf.ln(10)
    pdf.set_font("DejaVu", "B", 12)
    pdf.cell(0, 10, "CED графики", ln=True)
    pdf.set_font("DejaVu", size=12)
    for ds in ["300W", "Menpo"]:
        for exp in config.EXPERIMENTS:
            img_path = Path(config.RESULTS_DIR) / f"CED_{ds}_{exp['name']}.png"
            if img_path.exists():
                pdf.add_page()
                pdf.cell(0, 10, f"CED Curve for {ds} dataset - {exp['name']}", ln=True)
                pdf.image(str(img_path), x=10, y=20, w=pdf.w - 20)
    
    output_pdf = "experiment_report.pdf"
    pdf.output(str(output_pdf))
    print(f"Report saved to {output_pdf}")

if __name__ == "__main__":
    generate_report()
