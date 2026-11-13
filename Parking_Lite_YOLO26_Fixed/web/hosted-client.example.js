// Supply credentials at runtime to a trusted client. Never commit an API key
// or embed a shared server secret in a public website bundle.
export async function analyzeParking({baseUrl, token, file, quality = 'balanced'}) {
  const body = new FormData();
  body.append('file', file);
  body.append('quality', quality);
  body.append('include_image', 'true');
  const response = await fetch(new URL('/v1/analyze', baseUrl), {
    method: 'POST',
    headers: {Authorization: `Bearer ${token}`},
    body,
  });
  const result = await response.json();
  if (!response.ok) {
    const error = new Error(result.detail || `HTTP ${response.status}`);
    error.status = response.status;
    error.retryAfter = response.headers.get('Retry-After');
    throw error;
  }
  return {
    ...result,
    annotatedImageUrl: result.annotated_image_base64
      ? `data:${result.annotated_image_mime_type};base64,${result.annotated_image_base64}`
      : null,
  };
}
