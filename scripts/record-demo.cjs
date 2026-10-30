// npm install --no-save playwright@1.56.1; npx playwright install chromium
// Start the real app on port 8511 (TableToExcel) or 8766 (Curio). Requires ffmpeg/ffprobe.
const {chromium}=require('playwright');
const fs=require('fs');const path=require('path');const {spawnSync}=require('child_process');
const root=path.resolve(__dirname,'..');
async function main(){
 const mode='curio',repo=mode==='table'?'TableToExcel':'curio';
 const out=path.join(root,'media'),raw=path.join(root,'.demo-recording');fs.mkdirSync(raw,{recursive:true});
 const image=process.argv[2]||path.join(out,mode==='table'?'demo-input.png':'demo-input.jpg');
 if(path.resolve(image)!==path.join(out,mode==='table'?'demo-input.png':'demo-input.jpg')) fs.copyFileSync(image,path.join(out,mode==='table'?'demo-input.png':'demo-input.jpg'));
 const browser=await chromium.launch({headless:true});
 const context=await browser.newContext({viewport:{width:1200,height:850},recordVideo:{dir:raw,size:{width:1200,height:850}},colorScheme:'light'});
 const page=await context.newPage();const errors=[];page.on('pageerror',e=>errors.push(String(e)));
 await page.goto(mode==='table'?'http://127.0.0.1:8511':'http://127.0.0.1:8766',{waitUntil:'networkidle'});
 const start=Date.now();await page.waitForTimeout(600);
 if(mode==='table'){
   await page.locator('input[type=file]').setInputFiles(image);
   await page.getByRole('button',{name:'2 Â· Extract tables',exact:true}).waitFor({state:'visible'});
   await page.waitForTimeout(500);
   await page.getByRole('button',{name:'2 Â· Extract tables',exact:true}).click();
   await page.getByText('3 Â· Review and correct',{exact:true}).waitFor({timeout:120000});
   await page.getByText('3 Â· Review and correct',{exact:true}).scrollIntoViewIfNeeded();
   await page.getByRole('button',{name:'Download Excel workbook',exact:true}).waitFor();
   await page.locator('[data-testid="stDataFrame"]').scrollIntoViewIfNeeded();
   const grid=await page.locator('[data-testid="data-grid-canvas"]').boundingBox();
   await page.mouse.dblclick(grid.x+grid.width*.5,grid.y+36+35*2+17);
   await page.locator('textarea').last().waitFor({state:'visible',timeout:5000});
   await page.locator('textarea').last().fill('8');await page.keyboard.press('Enter');await page.waitForTimeout(1500);
   await page.screenshot({path:path.join(raw,'final.png')});
   await page.waitForTimeout(1400);
   const [download]=await Promise.all([page.waitForEvent('download'),page.getByRole('button',{name:'Download Excel workbook',exact:true}).click()]);
   await download.saveAs(path.join(raw,'tables.xlsx'));
 }else{
   await page.locator('#file').setInputFiles(image);
   await page.waitForFunction(()=>!document.querySelector('#results').classList.contains('hidden') && !document.querySelector('#status').textContent.includes('identifying'),{timeout:120000});
   await page.waitForTimeout(1000);
   await page.screenshot({path:path.join(raw,'final.png')});
   console.log(await page.locator('#verdict').innerText());
   await page.locator('#detail').scrollIntoViewIfNeeded();await page.waitForTimeout(1500);
   await page.evaluate(()=>scrollTo(0,0));
   await page.locator('nav button[data-tab=packs]').click();
   await page.waitForTimeout(1200);
   await page.locator('nav button[data-tab=identify]').click();
 }
 await page.waitForTimeout(Math.max(2200,8500-(Date.now()-start)));
 const video=page.video();await context.close();const source=await video.path();await browser.close();
 if(errors.length)throw Error(errors.join('\n'));
 const probe=JSON.parse(spawnSync('ffprobe',['-v','error','-show_entries','format=duration','-of','json',source],{encoding:'utf8'}).stdout);
 const duration=+probe.format.duration, speed=duration>9.5?duration/9:1;
 const filter=`[0:v]setpts=PTS/${speed},fps=12,scale=800:-1:flags=lanczos,split[a][b];[a]palettegen=stats_mode=diff[p];[b][p]paletteuse=dither=sierra2_4a`;
 const dest=path.join(out,'demo.gif');const result=spawnSync('ffmpeg',['-y','-i',source,'-filter_complex',filter,'-loop','0',dest],{stdio:'ignore'});if(result.status)throw Error('ffmpeg failed');
 if(fs.statSync(dest).size>=8000000)throw Error('GIF over 8 MB');
 fs.writeFileSync(path.join(out,'DEMO.md'),`# Real app recording\n\nRecorded ${new Date().toISOString()} from the unchanged local ${repo} application using Playwright and real inference/processing. No substituted UI or results. Playback speed: ${speed.toFixed(2)}Ã—; not a speed benchmark. 800 px wide, 12 fps, ${fs.statSync(dest).size} bytes.\n\n${mode==='table'?'The source is a generated printed-table test fixture (included as demo-input.png), passed through actual Tesseract OCR. The missed quantity 8 is corrected in the real editable grid before download. The XLSX download was saved and checked separately.':'The query is a public-domain Yellowstone quarter catalog photo from packs/us-mint-coins (included as demo-input.jpg). This is an in-catalog demonstration, not a held-out accuracy test. Actual DINOv2 inference and comparison results are shown; reference-photo attribution stays in the pack.'}\n`);
 console.log(JSON.stringify({repo,duration,speed,bytes:fs.statSync(dest).size}));
}
main().catch(e=>{console.error(e);process.exitCode=1});
