"""Double-blind review mode.

While the paper on the library is under double-blind review (ISMVL 2027), the
site must not reveal who built it or link to the analysis of the generation
models, which belongs to a separate paper. With ANONYMOUS_REVIEW on:

- author names, e-mail addresses and the institution are replaced by a notice
  (footer, imprint, About, Acknowledgements, Cite, generator page), and the
  release's authors become "Anonymous" in its BibTeX, in the API and in
  downloaded archives; the DOI and the Zenodo link stay, because reviewers
  need them to check the records (the Zenodo record's creators are set to
  anonymous there, by hand);
- the generation-model pages and links (Models, model counts, model names on
  implementations, the Generator entry in the menu) are hidden, and model
  names are left out of the API and of downloads.

Set ANONYMOUS_REVIEW = False after the review to restore all of it.
"""
ANONYMOUS_REVIEW = False

# What visitors see where names were
ANONYMOUS_NOTICE = 'Author information withheld for double-blind review.'
