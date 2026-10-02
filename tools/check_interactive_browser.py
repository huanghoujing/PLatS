import asyncio
import argparse
import json
from pathlib import Path
from playwright.async_api import async_playwright

parser=argparse.ArgumentParser(description='Exercise the released sample in a fresh interactive viewer session.')
parser.add_argument('--url',required=True,help='Complete server URL including #token')
parser.add_argument('--output',type=Path,required=True)
parser.add_argument('--chromium',help='Optional installed Chromium executable')
args=parser.parse_args()
OUT=args.output
OUT.mkdir(parents=True,exist_ok=True)
URL=args.url

async def main():
 async with async_playwright() as p:
  browser=await p.chromium.launch(headless=True,executable_path=args.chromium,args=['--enable-unsafe-swiftshader'])
  page=await browser.new_page(viewport={'width':1600,'height':1050})
  errors=[]
  page.on('pageerror',lambda error:errors.append(str(error)))
  await page.goto(URL)
  await page.wait_for_function('window.platsDiagnostics?.().shape?.length === 3',timeout=60000)
  await page.wait_for_function('window.platsDiagnostics?.().sheets.length === 1',timeout=60000)
  await page.locator('#auto').uncheck()
  async def click_voxel(axis,point,shift=False):
   await page.locator(f'#slider-{axis}').fill(str(point[axis]))
   await page.locator(f'#slider-{axis}').dispatch_event('input')
   pos=await page.evaluate('''({axis,point})=>{const d=window.platsDiagnostics(), b=d.boxes[axis], r=document.querySelector(`#slice-${axis}`).getBoundingClientRect();const map=[[2,1],[2,0],[0,1]][axis];return {x:r.x+b.x+(point[map[0]]+.5)*b.w/d.shape[map[0]],y:r.y+b.y+(point[map[1]]+.5)*b.h/d.shape[map[1]]};}''',{'axis':axis,'point':point})
   if shift:await page.keyboard.down('Shift')
   await page.mouse.click(pos['x'],pos['y'])
   if shift:await page.keyboard.up('Shift')
  points=[[155,117,153],[156,117,153],[155,118,153]]
  for axis,point in enumerate(points):await click_voxel(axis,point)
  d=await page.evaluate('window.platsDiagnostics()')
  assert d['sheets'][0]['points']==points,d
  assert d['webgl'], 'WebGL renderer failed'
  await click_voxel(2,[170,130,160],shift=True)
  d=await page.evaluate('window.platsDiagnostics()')
  assert d['slices']==[170,130,160] and len(d['sheets'][0]['points'])==3,d
  # Exercise exact one-point released-reference inference after mapping all axes.
  await page.locator('#undo').click();await page.locator('#undo').click()
  await page.locator('#decode').click()
  await page.wait_for_function('window.platsDiagnostics().sheets[0].revision > 0 && !window.platsDiagnostics().busy',timeout=120000)
  await page.locator('#slider-2').fill('153');await page.locator('#slider-2').dispatch_event('input')
  await page.screenshot(path=str(OUT/'viewer_one_point.png'),full_page=True)
  before=await page.locator('#slice-2').evaluate('(c)=>c.toDataURL()')
  await page.locator('#show-gt').check()
  after=await page.locator('#slice-2').evaluate('(c)=>c.toDataURL()')
  assert before!=after,'GT overlay did not change slice'
  await page.locator('#show-gt').uncheck()
  await page.locator('#threshold').fill('0.55');await page.locator('#threshold').dispatch_event('input');await page.locator('#threshold').dispatch_event('change')
  await page.locator('#decode').click()
  await page.wait_for_function('window.platsDiagnostics().sheets[0].revision >= 2 && !window.platsDiagnostics().busy',timeout=120000)
  assert '0.00s' in await page.locator('#timing').inner_text()
  await page.locator('#export').click()
  await page.wait_for_function('document.querySelector("#status").textContent === "NIFTI export complete"',timeout=120000)
  exported=await page.locator('#notice').inner_text()
  await page.locator('#notice').click()
  # Second sheet coexists with the first, with shared image context.
  await page.locator('#new-sheet').click();await page.locator('#auto').check()
  await click_voxel(0,[155,140,153])
  await page.wait_for_function('window.platsDiagnostics().sheets[1].revision > 0 && !window.platsDiagnostics().busy',timeout=120000)
  await page.screenshot(path=str(OUT/'viewer_two_sheets.png'),full_page=True)
  d=await page.evaluate('window.platsDiagnostics()')
  await page.reload()
  await page.wait_for_function('window.platsDiagnostics?.().sheets.length === 2 && window.platsDiagnostics().sheets.every(s=>s.revision>0)',timeout=60000)
  restored=await page.evaluate('window.platsDiagnostics()')
  assert [s['points'] for s in restored['sheets']]==[s['points'] for s in d['sheets']]
  assert not errors,errors
  report=dict(orthogonal_click_coordinates_exact=True,shift_click_navigation=True,webgl=True,
              one_point_decode=True,threshold_reuses_probability=True,gt_overlay=True,
              two_sheets=True,automatic_click_decode=True,browser_refresh_restores_decoded_sheets=True,nifti_export=exported,diagnostics=d,javascript_errors=errors)
  (OUT/'browser_validation.json').write_text(json.dumps(report,indent=2)+'\n')
  print(json.dumps(report,indent=2))
  await browser.close()

asyncio.run(main())
