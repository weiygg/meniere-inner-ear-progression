import fs from "node:fs/promises";
import path from "node:path";
import { SpreadsheetFile, Workbook } from "@oai/artifact-tool";


const projectRootArg = process.argv.find((value) => value.startsWith("--project-root="));
if (!projectRootArg) throw new Error("--project-root is required");
const projectRoot = path.resolve(projectRootArg.slice("--project-root=".length));
const auditPath = path.join(projectRoot, "data", "manifests", "clinical_workbook_audit.json");
const outputXlsx = path.join(projectRoot, "data", "manifests", "clinical_feature_schema.xlsx");
const outputJson = path.join(projectRoot, "data", "manifests", "clinical_feature_schema.json");
const outputMarkdown = path.join(projectRoot, "docs", "DATA_DICTIONARY.md");
const previewPath = path.join(projectRoot, "tmp", "clinical_schema_preview.png");
const fieldsPreviewPath = path.join(projectRoot, "tmp", "clinical_schema_fields_preview.png");

const audit = JSON.parse(await fs.readFile(auditPath, "utf8"));

const sheetMeta = {
  "丽水": {
    logicalName: "Lishui_ear_level",
    level: "ear-level; paired left/right rows",
    cohortRole: "primary development clinical sheet",
  },
  "浙二MD": {
    logicalName: "Z2_patient_questionnaire",
    level: "patient/questionnaire-level",
    cohortRole: "restricted linkage/questionnaire source; crosswalk unresolved",
  },
  "浙二": {
    logicalName: "Z2_ear_visit_level",
    level: "ear- and visit-level; paired left/right rows",
    cohortRole: "external clinical sheet; baseline and follow-up visits present",
  },
};

const definitions = {
  "": ["unnamed_column", "unclassified", "Blank source header; content role is not established.", "unknown", "exclude", "unresolved"],
  "ID": ["source_subject_or_visit_id", "identifier", "Site-local source subject/visit identifier.", "source-specific text/number", "identifier_only", "verified_header_only"],
  "编号": ["source_patient_number", "identifier", "Source-local patient number; sheet-to-sheet crosswalk is not confirmed.", "source-specific identifier", "identifier_only", "unresolved_crosswalk"],
  "姓名": ["patient_name", "direct_identifier", "Patient name.", "text", "exclude_phi", "direct_identifier"],
  "患者姓名": ["patient_name", "direct_identifier", "Patient name.", "text", "exclude_phi", "direct_identifier"],
  "电话": ["telephone", "direct_identifier", "Telephone number.", "text", "exclude_phi", "direct_identifier"],
  "病案号": ["medical_record_number", "direct_identifier", "Hospital medical-record number.", "identifier", "exclude_phi", "direct_identifier"],
  "出生日期": ["date_of_birth", "direct_identifier", "Date of birth.", "date", "exclude_phi", "direct_identifier"],
  "扫描日期": ["scan_date", "sensitive_date", "MRI examination date.", "date", "linkage_only", "restricted_date"],
  "起病时间": ["disease_onset_time", "clinical_timing", "Reported disease-onset time/date.", "date or duration; codebook needed", "candidate_covariate", "unresolved_definition"],
  "age": ["age_years", "demographic", "Age at the recorded assessment.", "years", "candidate_covariate", "unit_expected_not_signed"],
  "年龄": ["age_years", "demographic", "Age at the recorded assessment.", "years", "candidate_covariate", "unit_expected_not_signed"],
  "sex": ["sex", "demographic", "Recorded sex.", "coding requires source codebook", "candidate_covariate", "unresolved_coding"],
  "性别（男=1，女=2）": ["sex", "demographic", "Recorded sex; source header states male=1 and female=2.", "1=male; 2=female", "candidate_covariate", "coding_in_header"],
  "side": ["ear_side", "ear_identifier", "Ear side.", "L/R", "identifier_only", "coding_observed"],
  "MD1,SD2,其他3": ["diagnosis_group", "diagnosis", "Source diagnosis-group code.", "1=MD; 2=SD; 3=other per header; exact criteria needed", "cohort_eligibility", "criteria_unresolved"],
  "身高cm": ["height_cm", "anthropometry", "Height.", "cm", "candidate_covariate", "unit_in_header"],
  "体重kg": ["weight_kg", "anthropometry", "Weight.", "kg", "candidate_covariate", "unit_in_header"],
  "教育年限y": ["education_years", "demographic", "Years of education.", "years", "candidate_covariate", "unit_in_header"],
  "利手（右=0，左=1）": ["handedness", "demographic", "Handedness.", "0=right; 1=left", "candidate_covariate", "coding_in_header"],
  "CochEH": ["cochlear_endolymphatic_hydrops", "mri_hydrops", "Cochlear endolymphatic-hydrops assessment.", "ordinal coding; codebook required", "candidate_progression_biomarker", "unresolved_coding"],
  "VestEH": ["vestibular_endolymphatic_hydrops", "mri_hydrops", "Vestibular endolymphatic-hydrops assessment.", "ordinal coding; codebook required", "candidate_progression_biomarker", "unresolved_coding"],
  "VA": ["va_source_field", "imaging_anatomy", "Source field labelled VA; exact anatomical meaning and coding require confirmation.", "codebook required", "candidate_endotype_or_exclude", "unresolved_abbreviation"],
  "ES/ED": ["es_ed_source_field", "imaging_biomarker", "Source field labelled ES/ED; exact definition, direction, and coding require confirmation.", "codebook required", "candidate_biomarker", "unresolved_abbreviation"],
  "stage（AAO-HNS）": ["aao_hns_hearing_stage", "clinical_validator", "Recorded AAO-HNS hearing stage.", "ordinal stage; provenance/timing require confirmation", "validator_not_pebm_input", "provenance_unresolved"],
  "0.5kHZ": ["hearing_threshold_0_5khz", "audiometry", "Pure-tone hearing threshold at 0.5 kHz.", "dB HL expected; unit not confirmed", "candidate_progression_biomarker", "unit_unconfirmed"],
  "1kHZ": ["hearing_threshold_1khz", "audiometry", "Pure-tone hearing threshold at 1 kHz.", "dB HL expected; unit not confirmed", "candidate_progression_biomarker", "unit_unconfirmed"],
  "2kHZ": ["hearing_threshold_2khz", "audiometry", "Pure-tone hearing threshold at 2 kHz.", "dB HL expected; unit not confirmed", "candidate_progression_biomarker", "unit_unconfirmed"],
  "3kHZ": ["hearing_threshold_3khz", "audiometry", "Pure-tone hearing threshold at 3 kHz.", "dB HL expected; unit not confirmed", "candidate_progression_biomarker", "unit_unconfirmed"],
  "PTA": ["pta_aaohns_0_5_1_2_3khz", "audiometry", "Four-frequency mean of 0.5, 1, 2, and 3 kHz in the ear-level sheets.", "dB HL expected; recompute from frequencies", "candidate_progression_biomarker", "formula_audited_unit_unconfirmed"],
  "MMSE": ["mmse_total", "cognitive_scale", "Mini-Mental State Examination total score.", "score; version/range require codebook", "candidate_validator_or_covariate", "instrument_details_unconfirmed"],
  "HADS（≥12 +）": ["hads_source_score", "psychological_scale", "Hospital Anxiety and Depression Scale source field; header embeds a >=12 positive rule.", "score; subscale/total and threshold provenance require confirmation", "candidate_validator", "instrument_details_unconfirmed"],
  "DHI-F": ["dhi_functional", "vertigo_disability", "Dizziness Handicap Inventory functional subscore.", "score", "clinical_validator", "instrument_version_unconfirmed"],
  "DHI-E": ["dhi_emotional", "vertigo_disability", "Dizziness Handicap Inventory emotional subscore.", "score", "clinical_validator", "instrument_version_unconfirmed"],
  "DHI-P": ["dhi_physical", "vertigo_disability", "Dizziness Handicap Inventory physical subscore.", "score", "clinical_validator", "instrument_version_unconfirmed"],
  "DHI-T": ["dhi_total", "vertigo_disability", "Dizziness Handicap Inventory total score.", "score", "clinical_validator_not_irreversible_event", "instrument_version_unconfirmed"],
  "VADL": ["vadl_total", "activities_daily_living", "Vestibular Activities of Daily Living source score.", "score; version/range require codebook", "clinical_validator_not_irreversible_event", "instrument_details_unconfirmed"],
  "UCLA-DQ": ["ucla_dizziness_questionnaire", "vertigo_scale", "UCLA Dizziness Questionnaire source score.", "score; version/range require codebook", "clinical_validator", "instrument_details_unconfirmed"],
  "EEV": ["eev_source_score", "vertigo_scale", "Source field labelled EEV; instrument expansion and scoring require confirmation.", "score; codebook required", "clinical_validator", "unresolved_abbreviation"],
  "MoCA": ["moca_total", "cognitive_scale", "Montreal Cognitive Assessment total score.", "score; version/language require codebook", "candidate_validator_or_covariate", "instrument_details_unconfirmed"],
  "HIT-6": ["hit6_total", "headache_impact", "Six-item Headache Impact Test total score.", "score", "clinical_validator", "instrument_version_unconfirmed"],
  "MIDAS": ["midas_total", "migraine_disability", "Migraine Disability Assessment source score.", "score", "clinical_validator", "instrument_version_unconfirmed"],
  "PSQI": ["psqi_total", "sleep_quality", "Pittsburgh Sleep Quality Index source score.", "score", "clinical_validator", "instrument_version_unconfirmed"],
  "THI": ["thi_total", "tinnitus_handicap", "Tinnitus Handicap Inventory source score.", "score", "clinical_validator_not_irreversible_event", "instrument_version_unconfirmed"],
  "耳闷": ["aural_fullness", "symptom", "Aural fullness source field.", "binary/ordinal/text coding requires codebook", "clinical_validator_not_irreversible_event", "unresolved_coding"],
  "内耳检查结果": ["inner_ear_exam_result", "clinical_exam", "Inner-ear examination result field.", "text/categorical; codebook required", "source_review_only", "unresolved_coding"],
  "半规管显影": ["semicircular_canal_visibility", "imaging_exam", "Semicircular-canal visualization field.", "text/categorical; protocol and coding required", "candidate_imaging_feature", "unresolved_coding"],
  "前庭导水管显影": ["vestibular_aqueduct_visibility", "imaging_exam", "Vestibular-aqueduct visualization field.", "text/categorical; protocol and coding required", "candidate_endotype", "unresolved_coding"],
  "纯音听阈报告": ["pure_tone_audiogram_report", "source_document", "Pure-tone audiometry report indicator/reference.", "text/document reference", "linkage_or_source_review", "unresolved_content"],
  "声导抗": ["immittance_audiometry", "audiology_test", "Acoustic immittance/tympanometry source field.", "text/categorical; protocol and coding required", "candidate_validator", "unresolved_coding"],
  "前庭试验": ["vestibular_testing", "vestibular_test", "Vestibular-test source field.", "test type/result coding required", "candidate_progression_biomarker_or_validator", "unresolved_coding"],
  "其他": ["other_notes", "free_text", "Other clinical notes.", "free text", "exclude_until_adjudicated", "unstructured_sensitive_text"],
};

const phiHeaders = new Set(["姓名", "患者姓名", "电话", "病案号", "出生日期"]);
const dateHeaders = new Set(["扫描日期", "起病时间"]);
const rows = [];
for (const sheet of audit.sheets) {
  const meta = sheetMeta[sheet.sheet_name];
  for (let columnIndex = 0; columnIndex < sheet.headers.length; columnIndex += 1) {
    const header = sheet.headers[columnIndex] ?? "";
    const definition = definitions[header] ?? [
      `unmapped_${columnIndex + 1}`,
      "unclassified",
      "Header present in source; definition requires the study codebook.",
      "unknown",
      "exclude_until_adjudicated",
      "unmapped_header",
    ];
    const privacyClass = phiHeaders.has(header)
      ? "direct_identifier"
      : dateHeaders.has(header)
        ? "sensitive_date_or_quasi_identifier"
        : definition[1] === "identifier" || definition[1] === "ear_identifier"
          ? "restricted_identifier"
          : definition[1] === "free_text" || definition[1] === "source_document"
            ? "potentially_identifiable_content"
            : "clinical_metadata_no_values_exported";
    rows.push({
      source_sheet: sheet.sheet_name,
      logical_sheet: meta.logicalName,
      source_column_number: columnIndex + 1,
      original_header: header || "<blank>",
      standardized_name: definition[0],
      variable_group: definition[1],
      data_level: meta.level,
      description: definition[2],
      expected_unit_or_coding: definition[3],
      privacy_class: privacyClass,
      candidate_role: definition[4],
      definition_status: definition[5],
      github_content: "schema_only_no_patient_values",
    });
  }
}

const schemaPayload = {
  schema_version: 1,
  source: "protected clinical workbook header audit",
  workbook_sha256: audit.sha256,
  privacy: "schema only; zero patient rows or cell values",
  warning: "Candidate roles are provisional and do not freeze cohort, endpoint, or P-EBM eligibility decisions.",
  sheets: audit.sheets.map((sheet) => ({
    source_sheet: sheet.sheet_name,
    logical_sheet: sheetMeta[sheet.sheet_name].logicalName,
    data_level: sheetMeta[sheet.sheet_name].level,
    cohort_role: sheetMeta[sheet.sheet_name].cohortRole,
    source_data_rows_not_exported: sheet.data_rows,
    source_columns: sheet.columns,
  })),
  fields: rows,
};
await fs.writeFile(outputJson, `${JSON.stringify(schemaPayload, null, 2)}\n`, "utf8");

const workbook = Workbook.create();
const summarySheet = workbook.worksheets.add("Sheet_Summary");
const schemaSheet = workbook.worksheets.add("Clinical_Feature_Schema");

summarySheet.getRange("A1:F1").merge();
summarySheet.getRange("A1").values = [["Clinical workbook schema only / 临床表头字典"]];
summarySheet.getRange("A2:F2").merge();
summarySheet.getRange("A2").values = [["No patient rows or cell values are included. Candidate roles remain provisional."]];
summarySheet.getRange("A4:F4").values = [["Source sheet", "Logical sheet", "Data level", "Cohort role", "Source rows (not exported)", "Columns"]];
const summaryRows = schemaPayload.sheets.map((sheet) => [
  sheet.source_sheet,
  sheet.logical_sheet,
  sheet.data_level,
  sheet.cohort_role,
  sheet.source_data_rows_not_exported,
  sheet.source_columns,
]);
summarySheet.getRangeByIndexes(4, 0, summaryRows.length, 6).values = summaryRows;

const headers = Object.keys(rows[0]);
schemaSheet.getRangeByIndexes(0, 0, 1, headers.length).values = [headers];
schemaSheet.getRangeByIndexes(1, 0, rows.length, headers.length).values = rows.map((row) => headers.map((header) => row[header]));

for (const sheet of [summarySheet, schemaSheet]) {
  sheet.showGridLines = false;
  sheet.freezePanes.freezeRows(sheet === summarySheet ? 4 : 1);
}
summarySheet.getRange("A1:F1").format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF", size: 16 }, horizontalAlignment: "center", verticalAlignment: "center" };
summarySheet.getRange("A2:F2").format = { fill: "#D9EAF7", font: { italic: true, color: "#1F1F1F" }, wrapText: true };
summarySheet.getRange("A4:F4").format = { fill: "#5B9BD5", font: { bold: true, color: "#FFFFFF" }, wrapText: true };
summarySheet.getRange(`A5:F${4 + summaryRows.length}`).format = { wrapText: true, verticalAlignment: "top", borders: { preset: "inside", style: "thin", color: "#D9E2F3" } };
summarySheet.getRange("A1:F8").format.autofitRows();
summarySheet.getRange("A1:F8").format.autofitColumns();
summarySheet.getRange("A:A").format.columnWidth = 18;
summarySheet.getRange("B:B").format.columnWidth = 28;
summarySheet.getRange("C:D").format.columnWidth = 36;

schemaSheet.getRangeByIndexes(0, 0, 1, headers.length).format = { fill: "#1F4E78", font: { bold: true, color: "#FFFFFF" }, wrapText: true, verticalAlignment: "center" };
schemaSheet.getRangeByIndexes(1, 0, rows.length, headers.length).format = { wrapText: true, verticalAlignment: "top" };
schemaSheet.getRangeByIndexes(0, 0, rows.length + 1, headers.length).format.borders = { preset: "inside", style: "thin", color: "#E7E6E6" };
schemaSheet.getRangeByIndexes(0, 0, rows.length + 1, headers.length).format.autofitRows();
schemaSheet.getRangeByIndexes(0, 0, rows.length + 1, headers.length).format.autofitColumns();
const widths = [14, 27, 10, 24, 30, 24, 34, 52, 38, 32, 38, 30, 28];
for (let index = 0; index < widths.length; index += 1) schemaSheet.getRangeByIndexes(0, index, rows.length + 1, 1).format.columnWidth = widths[index];
schemaSheet.tables.add(`A1:M${rows.length + 1}`, true, "ClinicalFeatureSchemaTable").style = "TableStyleMedium2";

const inspect = await workbook.inspect({
  kind: "table",
  range: "Clinical_Feature_Schema!A1:M12",
  include: "values,formulas",
  tableMaxRows: 12,
  tableMaxCols: 13,
  maxChars: 6000,
});
console.log(inspect.ndjson);
const formulaErrors = await workbook.inspect({
  kind: "match",
  searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A",
  options: { useRegex: true, maxResults: 100 },
  summary: "clinical schema formula error scan",
});
console.log(formulaErrors.ndjson);

await fs.mkdir(path.dirname(previewPath), { recursive: true });
const preview = await workbook.render({ sheetName: "Sheet_Summary", range: "A1:F8", scale: 1.5, format: "png" });
await fs.writeFile(previewPath, new Uint8Array(await preview.arrayBuffer()));
const fieldsPreview = await workbook.render({ sheetName: "Clinical_Feature_Schema", range: "A1:M18", scale: 1.2, format: "png" });
await fs.writeFile(fieldsPreviewPath, new Uint8Array(await fieldsPreview.arrayBuffer()));
const xlsx = await SpreadsheetFile.exportXlsx(workbook);
await xlsx.save(outputXlsx);

const grouped = new Map();
for (const row of rows) {
  if (!grouped.has(row.source_sheet)) grouped.set(row.source_sheet, []);
  grouped.get(row.source_sheet).push(row);
}
const md = [
  "# Clinical workbook data dictionary",
  "",
  "This document exports **schema only** from the protected clinical workbook. It contains no patient rows, cell values, names, identifiers, dates, images, or report text.",
  "",
  "Candidate roles are provisional. Fields marked `unresolved` require the study codebook and must not be silently recoded or entered into P-EBM.",
  "",
  "## Sheet overview",
  "",
  "| Source sheet | Logical sheet | Level | Source rows not exported | Columns | Cohort role |",
  "|---|---|---|---:|---:|---|",
  ...schemaPayload.sheets.map((sheet) => `| ${sheet.source_sheet} | ${sheet.logical_sheet} | ${sheet.data_level} | ${sheet.source_data_rows_not_exported} | ${sheet.source_columns} | ${sheet.cohort_role} |`),
  "",
  "## 主要特征组",
  "",
  "- 人口学与协变量：年龄、性别、身高、体重、教育年限、利手。",
  "- MRI/内耳影像：耳蜗积水（CochEH）、前庭积水（VestEH）、VA、ES/ED、半规管及前庭导水管显影。",
  "- 听力学：0.5/1/2/3 kHz听阈、四频PTA、AAO-HNS听力分期、纯音听阈报告、声导抗。",
  "- 前庭与症状负担：前庭试验、DHI各维度、VADL、UCLA-DQ、EEV、耳闷。",
  "- 耳鸣、偏头痛、睡眠及认知：THI、HIT-6、MIDAS、PSQI、MMSE、MoCA、HADS。",
  "- 标识与时间字段：仅用于受控环境中的患者/耳/访视关联，不上传字段值，也不进入模型。",
  "",
  "## Privacy boundary",
  "",
  "- Direct identifiers (`姓名`, `患者姓名`, `电话`, `病案号`, `出生日期`) are listed only as field names and are excluded from modelling and GitHub data export.",
  "- Dates, source IDs, free text, document references, and examination narratives remain restricted; their values are not exported.",
  "- The uploaded workbook and JSON contain definitions only. They cannot reconstruct the clinical cohort.",
  "",
];
for (const [sheetName, sheetRows] of grouped) {
  md.push(`## ${sheetName}`, "", "| Original header | Standardized name | Group | Meaning | Unit/coding | Privacy | Candidate role | Definition status |", "|---|---|---|---|---|---|---|---|");
  for (const row of sheetRows) {
    const values = [row.original_header, row.standardized_name, row.variable_group, row.description, row.expected_unit_or_coding, row.privacy_class, row.candidate_role, row.definition_status]
      .map((value) => String(value).replaceAll("|", "\\|"));
    md.push(`| ${values.join(" | ")} |`);
  }
  md.push("");
}
while (md.at(-1) === "") md.pop();
await fs.writeFile(outputMarkdown, `${md.join("\n")}\n`, "utf8");

console.log(JSON.stringify({ fields: rows.length, sheets: schemaPayload.sheets.length, outputXlsx, outputJson, outputMarkdown, previewPath, fieldsPreviewPath }));
