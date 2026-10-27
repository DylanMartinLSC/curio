# Catalog source lists

A source list defines a pack in a form that's easy to review: which items it contains, their facts, and exactly which public images to use. A build script turns the source list into a pack under `packs/`. Don't edit generated packs by hand.

## `us-mint-coins.sources.json` → `packs/us-mint-coins`

This pack covers 86 US coin design types with 130 reference images:

- **Circulating coins:** Lincoln cents and Jefferson nickels, plus Roosevelt, Washington and Kennedy coins.
- **Programs:**
  - Westward Journey nickels and Lincoln Bicentennial cents;
  - 50 State, America the Beautiful and American Women quarters;
  - Native American and Presidential dollars.
- **Classic types:** Indian Head, Buffalo, Mercury, Standing Liberty, Walking Liberty, Franklin, Morgan and Peace.

Build or update it:

```bash
python scripts/build_us_mint_pack.py build
```

The build script needs only Python and Pillow. It works like this:

1. **Look up every file** listed in the source list through the Wikimedia Commons API.
2. **Check every file.** A file is accepted only if Commons marks it **public domain** *and* its author or credit field names the **United States Mint**. In practice the accepted files are either:
   - US Mint product images, or
   - Smithsonian National Numismatic Collection photos credited "US Mint (coin)", which are public domain as US government work.

   If any file fails the check, the build stops and prints why. You can pass `--allow-skips` to build anyway; failing files are left out.
3. **Download** a 640 px rendition of each file. If the source list says to, it keeps only the left or right half of a side-by-side image. Transparency is flattened, and downloads are cached in `.cache/commons/`.
4. **Write the pack:**
   - `pack.json` and `items.jsonl`;
   - `SOURCES.md`, the attribution for every image;
   - `sources.lock.json`, the Commons SHA-1 of every file used, so reviewers can see when an upstream file changed.

Rebuilding from unchanged sources gives byte-identical output.

### Accuracy

These numbers come from `curio eval`: 260 simulated phone photos, made by applying random rotation, framing, background, lighting, blur and JPEG compression to the reference images.

| Embedder | Top‑1 | Top‑3 |
|---|---|---|
| DINOv2 small (`--embedder dinov2`) | **99%** | 100% |
| DINOv2 base | 96% | 99% |
| SigLIP 2 base, 384 px | 96% | 99% |
| SigLIP 2 base, 224 px (Curio default) | 93% | 95% |

**The pack therefore declares `recommended_embedder: dinov2 (facebook/dinov2-small)`, which Curio picks automatically.** SigLIP's errors mostly sent blurry photos of program-quarter reverses to the generic "quarter obverse" items. These are simulated photos, so accuracy on real ones still needs measuring.

### Adding coins

Use discover mode to find verified images:

```bash
python scripts/build_us_mint_pack.py discover "Peace dollar reverse"
python scripts/build_us_mint_pack.py discover --category "American Women quarters"
python scripts/build_us_mint_pack.py discover "Delaware quarter" --show-rejected   # see why files fail
```

Then add an item to the source list:

```json
{"id": "quarter-atb-shawnee", "title": "America the Beautiful quarter: Shawnee National Forest, Illinois (2016)",
 "attributes": {"denomination": "25 cents", "years": "2016", "program": "America the Beautiful Quarters"},
 "files": {"reverse": ["File:...jpg"], "obverse": [{"file": "File:side-by-side.jpg", "crop": "left"}]},
 "notes": "Identification tips, key dates."}
```

Before opening a pull request:

1. **Rebuild** the pack.
2. **Look at the images.** File names on Commons are sometimes wrong.
3. **Run** `curio --packs packs/us-mint-coins eval` to check that accuracy didn't drop.

### Known gaps

- **Most 50 State quarter reverses are missing.** The classic Commons scans ("Delaware quarter, reverse side, 1999.jpg") are tagged public domain but record no author or source, so their provenance can't be verified and they fail the check on purpose. Only 6 states have Mint-credited images.
- **Some series are incomplete:**
  - 13 of the 56 America the Beautiful quarters;
  - 13 of the 40 Presidential dollars;
  - 5 of the Native American dollar years.
- **One item per design type,** not per date and mint mark. Dates and mint marks are small details that whole-image matching can't reliably read. Per-item `notes` list the key dates to check.
