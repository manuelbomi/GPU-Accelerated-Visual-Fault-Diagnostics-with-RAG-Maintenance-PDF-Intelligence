// Mirrors api/schemas.py exactly (field names included) so the wire format
// and the TypeScript types never drift out of sync silently.

export interface HealthResponse {
  status: string;
  image_search_ready: boolean;
  rag_ready: boolean;
  classifier_ready: boolean;
}

export interface SimilarImageResult {
  name: string;
  score: number;
  url: string;
}

export interface SearchResponse {
  query_filename: string;
  results: SimilarImageResult[];
}

export interface RelatedImage {
  name: string;
  url: string;
}

export interface RetrievedPassage {
  text: string;
  score: number;
}

export interface DiagnoseResponse {
  query_filename: string;
  predicted_label: string | null;
  predicted_confidence: number | null;
  top_images: RelatedImage[];
  related_docs: string[];
  retrieved_passages: RetrievedPassage[];
  synthesis: string | null;
}
