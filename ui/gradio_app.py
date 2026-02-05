from app.rag_app import Demo
import gradio as gr

rag_obj = Demo()

def chat(video_url, user_message, chat_history):
    if not video_url:
        return chat_history, "Please provide a YouTube URL."

    # Call backend
    answer = rag_obj.fetch_response(
        video_url=video_url,
        query=user_message,
        chat_history=chat_history
    )

    # Update history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": answer})


    return chat_history, ""

with gr.Blocks(title="YouTube RAG Chatbot") as demo:
    gr.Markdown("## Chat with a YouTube Video")

    video_url = gr.Textbox(
        label="YouTube Video URL",
        placeholder="https://www.youtube.com/watch?v=..."
    )

    chatbot = gr.Chatbot(label="Conversation")
    user_input = gr.Textbox(
        label="Your question",
        placeholder="Ask something about the video...",
    )

    state = gr.State([])

    send_btn = gr.Button("Send")

    send_btn.click(
        chat,
        inputs=[video_url, user_input, state],
        outputs=[chatbot, user_input],
    ).then(
        lambda x: x,
        inputs=state,
        outputs=chatbot
    )

demo.launch(share=True)