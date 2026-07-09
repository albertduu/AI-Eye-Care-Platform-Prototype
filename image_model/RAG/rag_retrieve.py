def retrieve_similar_images(image_path, top_k=5):
    return [
        {
            "rank": 1,
            "note": "FAISS retrieval placeholder connected successfully",
            "image_path": image_path,
            "similarity_status": "RAG module is connected. Full FAISS search can be added next."
        }
    ]