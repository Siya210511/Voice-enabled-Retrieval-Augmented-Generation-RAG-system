import json
import random

# Load MSMARCO sample (i will download this separately)
# For now, dummy data for testing

def prepare_chunked_data():
    """Create sample chunked dataset"""
    
    sample_docs = [
        {
            "title": "India Geography",
            "content": "India is a country in South Asia. It is the second most populous country in the world. The capital is New Delhi. India has 28 states and 8 union territories.",
            "source": "wikipedia"
        },
        {
            "title": "Banasthali Vidyapith",
            "content": "Banasthali Vidyapith is a private women's university located in Rajasthan, India. It was founded in 1935. The university is known for its focus on women's education.",
            "source": "institutional"
        },
        {
            "title": "Machine Learning Basics",
            "content": "Machine learning is a subset of artificial intelligence. It enables computers to learn from data without being explicitly programmed. Common algorithms include decision trees, neural networks, and SVM.",
            "source": "tech"
        },
        # Add 100+ more docs from MSMARCO...
    ]
    
    # Apply chunking
    chunked = []
    for doc in sample_docs:
        # Simple semantic chunking
        sentences = doc['content'].split('. ')
        chunk_text = ''
        
        for sentence in sentences:
            if len(chunk_text) + len(sentence) < 500:
                chunk_text += sentence + '. '
            else:
                if chunk_text:
                    chunked.append({
                        'text': chunk_text.strip(),
                        'metadata': {
                            'source': doc['source'],
                            'title': doc['title']
                        }
                    })
                chunk_text = sentence + '. '
        
        if chunk_text:
            chunked.append({
                'text': chunk_text.strip(),
                'metadata': {
                    'source': doc['source'],
                    'title': doc['title']
                }
            })
    
    # Save
    with open('chunked_msmarco.json', 'w') as f:
        json.dump(chunked, f)
    
    print(f"✅ Created {len(chunked)} chunks")

if __name__ == '__main__':
    prepare_chunked_data()
