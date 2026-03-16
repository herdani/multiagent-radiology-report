"""
Gradio UI — Radiology AI
Mandatory human-in-the-loop: every report requires radiologist review.
"""
import logging
import os
import gradio as gr
from dotenv import load_dotenv

load_dotenv("/home/moez/projects/radiology-ai/.env")
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from pipeline.preprocessor import preprocess
from agents.orchestrator import run_pipeline, resume_pipeline
from agents.image_analysis import run_with_xai


def process_scan(dicom_file, modality: str):
    if dicom_file is None:
        return (
            None, None, "", "", "", "",
            "⚠️ Please upload a DICOM file.",
            gr.update(visible=False),
        )

    try:
        result        = preprocess(dicom_path=dicom_file.name, output_dir="data/processed")
        png_path      = result["png_path"]
        anonymized_id = result["anonymized_id"]

        # run XAI + image analysis
        findings, xai_result = run_with_xai(
            png_path=png_path,
            anonymized_id=anonymized_id,
            modality=modality,
        )

        # run full pipeline — HIL always on
        state, thread_id = run_pipeline(
            png_path=png_path,
            anonymized_id=anonymized_id,
            modality=modality,
            hil=True,
        )

        report_text = state["report"].report_text if state.get("report") else ""
        validation  = state.get("validation")

        # QA summary
        if validation:
            qa_summary = f"{'✅ Passed' if validation.passed else '⚠️ Failed'} — Score: {validation.score}\n"
            if validation.issues:
                qa_summary += f"Issues: {', '.join(validation.issues)}\n"
            if validation.warnings:
                qa_summary += f"Warnings: {', '.join(validation.warnings)}\n"
        else:
            qa_summary = "QA not available"

        # scan info + pathology scores
        scan_info = (
            f"Modality: {modality} | "
            f"Urgency: {state['report'].urgency_level if state.get('report') else 'N/A'} | "
            f"ID: {anonymized_id}"
        )

        scores = xai_result.get("pathology_scores", {})
        if scores:
            top = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
            scan_info += "\n\nAI Detection Scores:\n"
            scan_info += "\n".join(f"  {k}: {v:.1%}" for k, v in top if v > 0.05)

        heatmap_path = xai_result.get("heatmap_path") if xai_result.get("heatmap_path") and \
            os.path.exists(xai_result.get("heatmap_path", "")) else None

        return (
            png_path,
            heatmap_path,
            report_text,
            qa_summary,
            scan_info,
            thread_id,
            "⏸️ Awaiting radiologist review — please review, edit if needed, then approve or reject.",
            gr.update(visible=True),   # HIL panel ALWAYS visible
        )

    except Exception as e:
        logger.error("UI error: %s", e, exc_info=True)
        return (
            None, None, "", "", "", "",
            f"❌ Error: {str(e)}",
            gr.update(visible=False),
        )


def approve_report(report_text: str, thread_id: str):
    if not report_text.strip():
        return "⚠️ No report to approve.", gr.update(visible=True)
    try:
        final_state = resume_pipeline(
            thread_id=thread_id,
            approved_report=report_text,
            approved=True,
        )
        return (
            f"✅ Report approved and saved. ID: {thread_id}",
            gr.update(visible=False),
        )
    except Exception as e:
        logger.error("Approve error: %s", e, exc_info=True)
        return f"❌ Error: {str(e)}", gr.update(visible=True)


def reject_report(thread_id: str):
    try:
        resume_pipeline(thread_id=thread_id, approved_report="", approved=False)
        return f"🔴 Report rejected. ID: {thread_id}", gr.update(visible=False)
    except Exception as e:
        return f"❌ Error: {str(e)}", gr.update(visible=True)


with gr.Blocks(title="Radiology AI") as demo:

    gr.Markdown("""
    # 🏥 Radiology AI — Multi-Agent Report Generation
    Upload a DICOM scan for AI-assisted radiology report generation.
    All patient data is anonymized on upload (GDPR compliant).
    **Every report requires radiologist review before finalization.**
    """)

    with gr.Row():
        # left column
        with gr.Column(scale=1):
            gr.Markdown("### Upload Scan")
            dicom_input    = gr.File(label="DICOM file (.dcm)", file_types=[".dcm"])
            modality_input = gr.Dropdown(
                choices=["CR", "MR", "CT", "DX", "US"],
                value="CR", label="Modality",
            )
            analyze_btn = gr.Button("🔍 Analyze Scan", variant="primary")

            gr.Markdown("### Scan")
            scan_image    = gr.Image(label="Original scan", type="filepath")

            gr.Markdown("### XAI Heatmap")
            heatmap_image = gr.Image(
                label="Grad-CAM (red = model focused here)",
                type="filepath",
            )
            scan_info = gr.Textbox(
                label="Scan info + detection scores",
                lines=8, interactive=False,
            )

        # right column
        with gr.Column(scale=1):
            gr.Markdown("### AI Generated Report")
            status_box  = gr.Textbox(label="Status", interactive=False)
            report_box  = gr.Textbox(
                label="Radiology report — review and edit before approving",
                lines=16, interactive=True,
            )
            qa_box      = gr.Textbox(label="QA validation", lines=4, interactive=False)
            thread_state = gr.State("")

            # HIL panel — always visible after analysis
            with gr.Group(visible=False) as hil_panel:
                gr.Markdown("### 👨‍⚕️ Radiologist Review")
                gr.Markdown(
                    "Review the AI-generated report above. "
                    "Edit directly in the text box if needed, then approve or reject. "
                    "**No report is finalized without your approval.**"
                )
                with gr.Row():
                    approve_btn = gr.Button("✅ Approve & Finalize", variant="primary")
                    reject_btn  = gr.Button("🔴 Reject & Discard", variant="stop")
                action_output = gr.Textbox(label="Result", interactive=False)

    analyze_btn.click(
        fn=process_scan,
        inputs=[dicom_input, modality_input],
        outputs=[
            scan_image, heatmap_image, report_box,
            qa_box, scan_info, thread_state,
            status_box, hil_panel,
        ],
    )
    approve_btn.click(
        fn=approve_report,
        inputs=[report_box, thread_state],
        outputs=[action_output, hil_panel],
    )
    reject_btn.click(
        fn=reject_report,
        inputs=[thread_state],
        outputs=[action_output, hil_panel],
    )

    gr.Markdown(
        "---\n"
        "*This system is an AI assistant for radiologists. "
        "All reports must be reviewed and approved by a qualified radiologist "
        "before clinical use. EU AI Act compliant — human oversight mandatory.*"
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        theme=gr.themes.Soft(),
    )
