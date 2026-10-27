"""Build the Section 3 reference report as editable Markdown and a six-page PDF using measured evidence."""
from pathlib import Path
import csv
import json
from html import escape
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Image, Table, TableStyle, KeepTogether
from PIL import Image as PILImage

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'section3/outputs'
REPORTS = ROOT/'section3/reports'


def rows(name):
    with (OUT/'tables'/f'{name}.csv').open(encoding='utf-8', newline='') as handle:
        return list(csv.DictReader(handle))


def content():
    s = json.loads((OUT/'summary.json').read_text(encoding='utf-8'))
    numeric = {r['feature']: r for r in rows('numeric_summary')}
    missing = {r['feature']: float(r['missing_percent']) for r in rows('missingness')}
    r = s['strongest_numeric_correlation']
    n = s['development_encounters']
    p = lambda text: ('p', text)
    h = lambda text: ('h', text)
    fig = lambda name, caption, width=6.9: ('figure', name, caption, width)
    tab = lambda headers, body, widths: ('table', headers, body, widths)
    pages = [
        [('title', 'Hospital Readmission Data Preparation'),
         ('subtitle', 'Section 3 reference report'),
         p('This analysis prepares historical diabetes hospital encounters for a later readmission study. Its main findings are pronounced class imbalance, substantial missingness in a few categorical fields, and skewed prior-utilization counts. The preprocessing preserves valid unusual encounters, separates patients before learning transformations, and produces interpretable selected inputs alongside a PCA alternative. This document covers exploratory analysis and preprocessing only.'),
         h('Analysis population'),
         p('The UCI release contains 101,766 encounters from 71,518 patients at 130 US hospitals during 1999-2008 [1]. A row is an encounter, so a patient can contribute multiple rows. The source has 47 candidate inputs, two identifiers and one recorded outcome. Eight inputs are numerical counts or durations; the other 39 are treated as categorical.'),
         tab(['Population', 'Encounters', 'Patients'],
             [['Original release', '101,766', '71,518'], ['Eligible cohort', '99,343', '69,990'],
              ['Development for this analysis', f'{n:,}', f"{s['development_patients']:,}"],
              ['Reserved calibration', '19,746', '14,025'], ['Reserved test', '19,927', '14,005']], [3.7, 1.6, 1.6]),
         p('Exclude 2,423 encounters with death or hospice discharge codes 11, 13, 14, 19, 20 or 21. A five-fold grouped assignment with seed 459 places three folds in development and one each in calibration and test. No patient crosses partitions. Every distribution, association, learned transformation and feature-selection result below uses development data only.'),
         fig('01_population_missingness', f"Figure 1. Development class distribution and missingness. There are {s['development_positives']:,} recorded early readmissions ({s['development_positive_rate']:.2%}). The eligible full-cohort rate is 11.39%; the denominators differ."),
         p('The positive outcome is the released <30 category; >30 and NO form the negative class. NO means no recorded readmission in this release, not verified absence everywhere. Prediction is intended at discharge, when completed-stay variables are available.')],

        [h('Distributions and class imbalance'),
         p(f"An always-negative decision would match {1-s['development_positive_rate']:.2%} of development labels while identifying no early readmissions. Accuracy alone would therefore be inadequate for later model evaluation. The class proportions are preserved throughout this section."),
         fig('02_distributions_boxplots', 'Figure 2. Four representative distributions and comparisons by outcome. Histograms include all observations and use log count axes. Boxplots use 5th-95th percentile whiskers, with outside points hidden for readability; no rows are removed.'),
         tab(['Numerical input', 'Median', '99th pct', 'Maximum', 'Zero %'],
             [[label, f"{float(numeric[name]['50%']):g}", f"{float(numeric[name]['99%']):g}",
               f"{float(numeric[name]['max']):g}", f"{float(numeric[name]['zero_percent']):.1f}"]
              for name, label in [('time_in_hospital', 'Length of stay in days'), ('num_medications', 'Medications'),
                                  ('number_inpatient', 'Prior inpatient visits'), ('number_emergency', 'Prior emergency visits')]],
             [2.9, 1, 1, 1, 1]),
         p('Length of stay has a median of four days, whereas medication counts extend to 79. Emergency visits have a median of zero, a 99th percentile of three and a maximum of 76. A mean alone would conceal this concentration at zero and the long tail. Standardization changes scale, not skewness; the tail remains in the prepared numerical inputs.'),
         p('The early-readmission group has a higher upper quartile for prior inpatient visits (two versus one), but both groups have a median of zero. Medication medians are close (16 versus 15). The overlapping boxplots discourage a claim that a single variable cleanly separates outcomes. Full summaries of all eight numerical fields are supplied in numeric_summary.csv.')],

        [h('Relationships and interpretation'),
         fig('03_correlations', 'Figure 3. Spearman correlations between the eight numerical inputs, using development encounters only.', width=6.45),
         p(f"The largest absolute off-diagonal correlation is between length of stay and medication count (rho = {r['spearman_rho']:.3f}). No pair reaches the predeclared absolute-rho threshold of 0.80. This gives no evidence for removing one of these numeric fields solely because of very strong pairwise rank correlation. It does not rule out nonlinear dependence or redundancy involving categorical variables."),
         p('Categorical associations provide another descriptive view. The HbA1c category None contains 49,453 encounters and has an observed early-readmission rate of 11.69%; Norm contains 2,968 encounters with a rate of 10.58%. These are unadjusted encounter-level rates. Testing practices, illness severity and repeat encounters may affect the comparison; it cannot establish that ordering a test causes a change in readmission.'),
         p('Small categories require special care. The tables retain counts next to rates so that a high percentage from a handful of encounters is not mistaken for a stable pattern. No inferential significance or causal effect is claimed from these exploratory comparisons.')],

        [h('Cleaning missing values and unusual records'),
         p(f"Only ? is parsed as missing. Weight is {missing['weight']:.2f}% missing in development and is removed by the rule excluding inputs above 95% missingness. Medical specialty ({missing['medical_specialty']:.2f}%) and payer code ({missing['payer_code']:.2f}%) remain, with missingness encoded explicitly. The lab category None means not performed; it occurs in 82.88% of HbA1c values and 94.90% of glucose-test values and is preserved."),
         p('Four medication columns are constant in development after cleaning: acetohexamide, examide, citoglipton and metformin-rosiglitazone. Together with weight, their removal reduces 47 candidate inputs to 42. These are training-learned exclusions rather than assumptions about every possible future cohort. Diagnosis codes are mapped to fixed broad ICD-9 groups, retaining missing values and reducing sparse categorical detail.'),
         h('Imputation scaling and encoding'),
         p('The numerical branch uses median imputation followed by StandardScaler: subtract the training mean and divide by the training population standard deviation. The development numerical inputs contain zero missing cells, so median imputation is a robustness rule and changes no values in this run. The audit confirms means approximately zero and population standard deviations approximately one after scaling.'),
         p('The categorical branch fills missing values with Missing and uses one-hot encoding. Categories with fewer than 25 training occurrences are pooled. Unseen categories map to that pooled column where it exists; otherwise the feature has an all-zero block. Age bands and integer-coded admission/discharge categories are not treated as measured quantities. Encoding all 42 retained inputs produces 216 numerical columns.'),
         h('Outlier handling'),
         tab(['Screen', 'Flagged encounters', 'Action'],
             [['Any 1.5 IQR fence', f"{s['iqr_any_flagged']:,} ({s['iqr_any_flagged']/n:.2%})", 'Retain for analysis'],
              ['Isolation Forest', f"{s['isolation_flagged']:,} ({s['isolation_flagged']/n:.2%})", 'Retain for analysis'],
              ['Removed as outliers', '0', 'No automatic deletion']], [2.6, 2, 2.3]),
         p('Outpatient and emergency visit counts have IQR zero, so the IQR rule flags every nonzero count. Isolation Forest provides a complementary screen on log1p-transformed, imputed and standardized numerical values, using 200 trees and a predeclared 3% flag fraction [4]. Its flagged records have median prior emergency and inpatient counts of two each, compared with zero in the unflagged group. Flags indicate unusual utilization, not proven errors. Without adjudicated error labels, deleting these records could remove useful risk variation.'),
         p('No augmentation, oversampling or synthetic rows are used. The original class prevalence is retained for this analysis. Any future resampling must occur inside training folds; applying it before splitting can leak information or create unrealistic mixtures of clinical categories.')],

        [h('Independent feature selection and dimensionality reduction'),
         p('Mutual information ranks each whole cleaned input against the development target before one-hot encoding. Counts and categories are treated as discrete; missing values have their own code. This is a supervised filter independent of classifier fitting [2]. The predeclared top-20 option retains 20 raw inputs and yields 147 encoded columns.'),
         fig('04_selection_pca', 'Figure 4. Independent mutual-information rankings and cumulative variance for the separate PCA representation. Both are fitted using development encounters only.'),
         p('Prior inpatient visits rank first (0.01148 nats), followed by discharge disposition (0.00576) and emergency visits (0.00239). The scores describe estimated univariate dependence, not causal influence or an accuracy contribution. MI can favor high-cardinality variables and miss interactions; 20 features is a bounded comparison choice rather than a demonstrated optimum.'),
         p(f"PCA is applied to the full 216-column encoded representation, with standardized numerical inputs and binary categorical indicators. The input is centered; the one-hot indicators are not separately scaled to unit variance. Full SVD retains {s['pca_components']} components explaining {s['pca_retained_variance']:.2%} of development variance [3]. The first two components explain only {s['pca_first_two_variance']:.2%}, so a 2D projection would omit most variation."),
         p('PCA does not use labels and produces combinations of features, reducing direct clinical interpretability. Its retained variance is not evidence of better readmission prediction. The PCA outputs are a separate diagnostic alternative and do not replace the proposal classifier inputs. Any later cross-validation must refit cleaning, imputation, encoding, MI and any PCA inside each training fold.')],

        [h('Reproducibility and limitations'),
         p('The implementation records source and data hashes, package versions, patient partitions, feature decisions, encoding widths and matrix row order. Six focused tests check training-only imputation/scaling, selection before encoding, zero-IQR behavior, sparse-matrix reproduction, PCA reproduction and alignment of the audit with development data. The executed notebook demonstrates the same results and an unseen-category transform without refitting.'),
         ('code', 'python -m section3.run\npython -m pytest tests/test_section3.py -q\npython -m section3.build_notebook --execute'),
         p('Run these commands from the repository root after installing requirements.txt. The first command automatically obtains the public data if necessary, prepares the fixed patient split and runs this section only. Matching outputs are reused after checking their hashes; --force recomputes this section. The prepared sparse matrix, PCA scores and fitted transformers are regenerated locally and are not required to inspect the saved notebook.'),
         h('Limits of the evidence'),
         p('This is a retrospective analysis of historical US encounters, not evidence of performance in current Canadian hospitals. Missingness may reflect selective documentation and care processes. The release does not support reliable external-hospital or chronological validation because the required hospital identifiers and usable dates are absent. Repeated encounters contribute correlated observations. Label capture can miss readmissions elsewhere. These limitations persist after technical cleaning.'),
         p('The four selected figures cover class balance and missingness, distributions and outcome comparisons, numerical correlations, and feature selection with PCA. Detailed tables provide checks without crowding the report with every generated display. Classifier results, calibration, clustering comparisons and clinical deployment claims are outside this section.'),
         h('References'),
         ('reference', '[1] Clore, Cios, DeShazo and Strack. Diabetes 130 US Hospitals for Years 1999-2008. UCI Machine Learning Repository. Dataset DOI 10.24432/C5230J. CC BY 4.0.', 'https://doi.org/10.24432/C5230J'),
         ('reference', '[2] scikit-learn 1.7.2. mutual_info_classif documentation. Estimating dependence between discrete features and a discrete target.', 'https://scikit-learn.org/1.7/modules/generated/sklearn.feature_selection.mutual_info_classif.html'),
         ('reference', '[3] scikit-learn 1.7.2. PCA documentation. Centering, SVD and explained variance.', 'https://scikit-learn.org/1.7/modules/generated/sklearn.decomposition.PCA.html'),
         ('reference', '[4] scikit-learn 1.7.2. IsolationForest documentation. Anomaly scoring and contamination setting.', 'https://scikit-learn.org/1.7/modules/generated/sklearn.ensemble.IsolationForest.html'),
         p('All numerical results and figures in this reference come from the saved Section 3 run. The root README and STUDY_GUIDE.md explain the implementation for team review. Use this reference to understand and verify the work before writing the team-authored final report.')]
    ]
    return pages


def build():
    REPORTS.mkdir(parents=True, exist_ok=True)
    pages = content()
    styles = getSampleStyleSheet()
    styles.add(ParagraphStyle('ReportTitle', fontName='Helvetica-Bold', fontSize=20, leading=24, textColor=colors.black, spaceAfter=6))
    styles.add(ParagraphStyle('ReportSubtitle', fontName='Helvetica', fontSize=11, leading=14, textColor=colors.black, spaceAfter=13))
    styles.add(ParagraphStyle('ReportHeading', fontName='Helvetica-Bold', fontSize=13, leading=16, spaceBefore=10, spaceAfter=7, keepWithNext=True, textColor=colors.black))
    styles.add(ParagraphStyle('ReportBody', fontName='Helvetica', fontSize=10.3, leading=14, spaceAfter=8))
    styles.add(ParagraphStyle('ReportCaption', fontName='Helvetica', fontSize=9, leading=12, spaceAfter=9))
    styles.add(ParagraphStyle('ReportCell', fontName='Helvetica', fontSize=9.2, leading=12))
    styles.add(ParagraphStyle('ReportCode', fontName='Courier', fontSize=9, leading=13, spaceBefore=3, spaceAfter=9))
    story, markdown = [], []
    for page_number, blocks in enumerate(pages):
        if page_number:
            story.append(PageBreak())
            markdown.append('\n<!-- page break -->\n')
        for block in blocks:
            kind = block[0]
            if kind in {'title', 'subtitle', 'h', 'p'}:
                style = {'title': 'ReportTitle', 'subtitle': 'ReportSubtitle', 'h': 'ReportHeading', 'p': 'ReportBody'}[kind]
                story.append(Paragraph(escape(block[1]), styles[style]))
                markdown.append(('# ' if kind == 'title' else '## ' if kind == 'h' else '') + block[1] + '\n')
            elif kind == 'figure':
                target = OUT/'figures'/f'{block[1]}.png'
                with PILImage.open(target) as img:
                    width, height = img.size
                actual_width = block[3]*inch
                story.append(KeepTogether([Image(str(target), width=actual_width, height=actual_width*height/width), Spacer(1, 5), Paragraph(escape(block[2]), styles['ReportCaption'])]))
                markdown.append(f'![{block[2]}](../outputs/figures/{block[1]}.png)\n\n{block[2]}\n')
            elif kind == 'table':
                headers, body, widths = block[1:]
                cells = [[Paragraph(escape(str(c)), styles['ReportCell']) for c in row] for row in [headers, *body]]
                table = Table(cells, colWidths=[w*inch for w in widths], repeatRows=1, hAlign='LEFT')
                table.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e8eef1')),
                    ('GRID', (0,0), (-1,-1), .5, colors.HexColor('#D9D9D9')), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
                    ('LEFTPADDING', (0,0), (-1,-1), 7), ('RIGHTPADDING', (0,0), (-1,-1), 7),
                    ('TOPPADDING', (0,0), (-1,-1), 6), ('BOTTOMPADDING', (0,0), (-1,-1), 6)]))
                story += [table, Spacer(1, 10)]
                markdown += ['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |']
                markdown += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in body]
                markdown.append('')
            elif kind == 'code':
                story.append(Paragraph(escape(block[1]).replace('\n', '<br/>'), styles['ReportCode']))
                markdown.append('```bash\n'+block[1]+'\n```\n')
            elif kind == 'reference':
                story.append(Paragraph(escape(block[1])+f' <link href="{escape(block[2], quote=True)}" color="#246b83">Source</link>', styles['ReportCaption']))
                markdown.append(block[1]+f' [Source]({block[2]})\n')
    def footer(canvas, doc):
        canvas.saveState()
        canvas.setFont('Helvetica', 8.5)
        canvas.setFillColor(colors.HexColor('#555555'))
        canvas.drawString(.7*inch, .38*inch, 'GenoMining | Section 3 reference report')
        canvas.drawRightString(7.8*inch, .38*inch, str(doc.page))
        canvas.restoreState()
    target = REPORTS/'section3_reference_report.pdf'
    document = SimpleDocTemplate(str(target), pagesize=(8.5*inch, 11*inch), leftMargin=.7*inch,
                                 rightMargin=.7*inch, topMargin=.6*inch, bottomMargin=.65*inch,
                                 title='Hospital Readmission Data Preparation', author='GenoMining project reference')
    document.build(story, onFirstPage=footer, onLaterPages=footer)
    (REPORTS/'section3_reference_report.md').write_text('\n'.join(markdown), encoding='utf-8')
    print('Created reference PDF and editable Markdown in section3/reports')


if __name__ == '__main__':
    build()
