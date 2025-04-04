The splitting of words between these files is somewhat subjective, but roughly as follows:

 - `acronyms.txt`: Acronyms or initialisms (self-explanatory).
 - `custom_misc.txt`: Things that don't fit in the other files, including exceptions for American English spellings where required
 - `library_terms.txt`: Terms from third-party code, such as package, function and argument names.
     Often these are shortenings or concatenations of English words, like `diag` or `lengthscale`.
     If cspell complains about a code term that is _not_ from third-party code, change the code term rather than adding
     it to the dictionary!
 - `maths_terms.txt`: Maths terms that are missing from the default dictionary, like `heteroskedastic`.
 - `people.txt`: Names of people; largely these are authors cited in `references.bib`.
