// Shared helpers for the pages that extend base.html.

// Every endpoint answers with JSON. When the session has expired the request is
// redirected to the sign-in page and the body is HTML, so response.json() fails
// on the first character of "<!doctype html>" and reports "Unexpected token
// '<'" -- the real cause, being signed out, never reaches the user. Read every
// response through this instead.
async function readJson(response) {
    const type = response.headers.get('content-type') || '';
    if (!type.includes('application/json')) {
        const signedOut = response.status === 401 || response.status === 403
                          || (response.redirected && /\/login/.test(response.url));
        throw new Error(signedOut
            ? 'your session has expired — please sign in again'
            : `the server answered with ${response.status} instead of JSON`);
    }
    return response.json();
}
