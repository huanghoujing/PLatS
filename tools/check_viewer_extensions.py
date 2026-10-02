import argparse
import asyncio,json
import numpy as np
import tifffile
from pathlib import Path
from playwright.async_api import async_playwright

parser=argparse.ArgumentParser(description='Test viewer extensions on a fresh released sample_00860 session.')
parser.add_argument('--url',required=True,help='Complete server URL including token')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--sample-root',type=Path,default=Path('examples/sample_00860'))
parser.add_argument('--chromium',help='Optional Chromium executable')
args=parser.parse_args()
RUN=args.output.resolve();(RUN/'fixtures').mkdir(parents=True,exist_ok=True)
URL=args.url
for src,dst in [('image.npy','sample_00860.tif'),('gt_instances.npy','reference_00860.tif')]:
    volume=np.load(args.sample_root/src).transpose(2,1,0)
    tifffile.imwrite(RUN/'fixtures'/dst,volume,photometric='minisblack',compression='deflate',metadata={'axes':'ZYX'})

async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,executable_path=args.chromium,args=['--enable-unsafe-swiftshader'])
  page=await browser.new_page(viewport={'width':1700,'height':1100})
  errors=[];page.on('pageerror',lambda e:errors.append(str(e)))
  await page.goto(URL)
  await page.wait_for_function('window.platsDiagnostics?.().sheets.length === 1',timeout=90000)
  if not (await page.evaluate('window.platsDiagnostics()'))['automaticInstances']:
   await page.locator('#automatic').click()
  await page.wait_for_function('window.platsDiagnostics?.().automaticInstances === 14 && !window.platsDiagnostics().loading',timeout=90000)
  await page.locator('#automatic-overlay').uncheck()
  await page.locator('#automatic-3d').uncheck()
  await page.locator('#reference-id').select_option('9')
  await page.locator('#reference-3d').check()
  await page.wait_for_function('window.platsDiagnostics().referenceSurface',timeout=60000)
  await page.locator('#auto').uncheck()
  await page.locator('#slider-2').fill('153');await page.locator('#slider-2').dispatch_event('input')
  before=await page.locator('#slice-2').bounding_box()
  await page.locator('#layout').select_option('2')
  await page.wait_for_timeout(300)
  after=await page.locator('#slice-2').bounding_box()
  assert after['height']>before['height']*1.5,(before,after)
  pos=await page.evaluate('''()=>{const d=window.platsDiagnostics(),b=d.boxes[2],r=document.querySelector('#slice-2').getBoundingClientRect();return {x:r.x+b.x+155.5*b.w/320,y:r.y+b.y+117.5*b.h/320};}''')
  await page.mouse.click(pos['x'],pos['y']);await page.locator('#decode').click()
  await page.wait_for_function('window.platsDiagnostics().sheets[0].revision>0&&!window.platsDiagnostics().busy',timeout=90000)
  assert (await page.evaluate('window.platsDiagnostics()'))['sheets'][0]['points']==[[155,117,153]]
  await page.screenshot(path=str(RUN/'reference_and_prediction.png'),full_page=True)
  await page.locator('#show-3d-panel').uncheck()
  assert not await page.locator('#panel-3').is_visible()
  await page.screenshot(path=str(RUN/'single_slice.png'),full_page=True)
  await page.locator('#show-3d-panel').check()
  await page.locator('summary').click()
  await page.locator('#image-path').fill(str(RUN/'fixtures/sample_00860.tif'))
  await page.locator('#gt-path').fill(str(RUN/'fixtures/reference_00860.tif'))
  await page.locator('#load').click()
  await page.wait_for_function('window.platsDiagnostics().generation===2&&!window.platsDiagnostics().loading',timeout=90000)
  await page.locator('#reference-3d').uncheck()
  await page.locator('#automatic-overlay').check();await page.locator('#automatic-3d').check()
  await page.locator('#automatic').click()
  await page.wait_for_function('window.platsDiagnostics().automaticInstances===14&&!window.platsDiagnostics().loading',timeout=180000)
  await page.locator('#automatic-id').select_option('5')
  await page.wait_for_function('window.platsDiagnostics().automaticSurface',timeout=60000)
  await page.screenshot(path=str(RUN/'automatic_instances.png'),full_page=True)
  await page.locator('#export-format').select_option('tiff')
  await page.locator('#export').click()
  await page.wait_for_function('document.querySelector("#status").textContent==="Export complete"',timeout=90000)
  output=await page.locator('#notice').inner_text();(RUN/'tiff_export_path.txt').write_text(output.splitlines()[-1])
  await page.locator('#notice').click()
  await page.locator('#input-source').select_option('zarr')
  remote=dict(url='https://vesuvius-challenge-open-data.s3.amazonaws.com/PHerc0139/volumes/20250728140407-9.362um-1.2m-113keV-masked.zarr',origin_xyz=[3612,3942,4572])
  await page.locator('#zarr-url').fill(remote['url'])
  for a,v in zip('xyz',remote['origin_xyz']):await page.locator(f'#start-{a}').fill(str(v))
  await page.locator('#load').click()
  await page.wait_for_function('window.platsDiagnostics().generation===3&&!window.platsDiagnostics().loading',timeout=90000)
  assert not await page.locator('#reference-3d').is_enabled()
  await page.locator('#layout').select_option('three')
  await page.screenshot(path=str(RUN/'official_zarr_patch.png'),full_page=True)
  # Real remote inference from the center of the preselected reference region.
  await page.locator('#auto').check()
  pos=await page.evaluate('''()=>{const d=window.platsDiagnostics(),b=d.boxes[2],r=document.querySelector('#slice-2').getBoundingClientRect();return {x:r.x+b.x+160.5*b.w/320,y:r.y+b.y+160.5*b.h/320};}''')
  await page.mouse.click(pos['x'],pos['y'])
  await page.wait_for_function('window.platsDiagnostics().sheets[0].revision>0&&!window.platsDiagnostics().busy',timeout=90000)
  await page.screenshot(path=str(RUN/'official_zarr_prediction.png'),full_page=True)
  await page.locator('#export').click()
  await page.wait_for_function('document.querySelector("#status").textContent==="Export complete"',timeout=90000)
  remote_output=await page.locator('#notice').inner_text();(RUN/'remote_export_path.txt').write_text(remote_output.splitlines()[-1])
  assert not errors,errors
  report=dict(reference_3d_with_prediction=True,enlarged_slice_height_ratio=after['height']/before['height'],
    single_slice_click_exact=True,hide_3d_panel=True,challenge_tiff_load=True,automatic_button_14_instances=True,
    automatic_tiff_export=True,official_zarr_xyz_load=True,remote_click_decode=True,remote_tiff_export=True,javascript_errors=errors)
  (RUN/'browser_extensions_validation.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report,indent=2));await browser.close()
asyncio.run(main())
