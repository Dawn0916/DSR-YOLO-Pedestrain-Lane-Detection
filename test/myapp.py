import streamlit as st
import plotly.express as px
from video_handler import VideoProcessor
#from streamlit_webrtc import webrtc_streamer, RTCConfiguration
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode, RTCConfiguration

# WebRTC Configuration (necessary for video streaming)
RTC_CONFIGURATION = RTCConfiguration({
    "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
})

def main():
    st.title("Real-Time Object Tracking with YOLOv8")

    with st.sidebar:
        st.title("Configuration")
        st.subheader("Video Streaming Configuration")
        
        # Streamlit-webrtc streaming component
        webrtc_ctx = webrtc_streamer(
            key="example",
            #mode="SENDRECV",
            mode=WebRtcMode.SENDRECV, 
            #rtc_configuration=VideoProcessor,
            #video_transformer_factory=VideoHandler,
            video_processor_factory=VideoProcessor,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True
        )

    if webrtc_ctx.video_transformer:
        # Get tracking results DataFrame from the VideoHandler class
        tracking_results = webrtc_ctx.video_transformer.get_tracking_results()

        if not tracking_results.empty:
            # Display tracking results
            st.write(tracking_results)

            # Plot the class counts over time using Plotly
            group_data = tracking_results[["frame_no", "class_name"]].groupby(
                ["frame_no", "class_name"]).size().reset_index(name="count")
            final_df = group_data.pivot(index="frame_no", columns="class_name", values="count").fillna(0)

            fig = px.line(final_df,
                          x=final_df.index,
                          y=tracking_results.class_name.unique(),
                          title='Class counts for each frame')

            fig.update_xaxes(title_text='Frame number')
            fig.update_yaxes(title_text='Counts')

            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
