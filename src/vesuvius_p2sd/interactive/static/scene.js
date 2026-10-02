import * as THREE from "three";
import { OrbitControls } from "./vendor/three/OrbitControls.js";

export class SurfaceView {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color("#0b1521");
    this.camera = new THREE.PerspectiveCamera(42, 1, 0.1, 5000);
    this.renderer = new THREE.WebGLRenderer({ antialias: true });
    this.renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    container.append(this.renderer.domElement);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.needsRender = true;
    this.controls.addEventListener("change", () => {
      this.needsRender = true;
    });
    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.meshes = new Map();
    this.markers = new THREE.Group();
    this.root.add(this.markers);
    this.planes = [];
    this.shape = [320, 320, 320];
    this.scene.add(new THREE.HemisphereLight(0xd5eaff, 0x374459, 2.3));
    for (const [x, y, z] of [
      [500, -500, 600],
      [-400, 300, -200],
    ]) {
      const light = new THREE.DirectionalLight(0xffffff, 2);
      light.position.set(x, y, z);
      this.scene.add(light);
    }
    new ResizeObserver(() => this.resize()).observe(container);
    this.renderer.setAnimationLoop(() => {
      this.controls.update();
      if (this.needsRender) {
        this.renderer.render(this.scene, this.camera);
        this.needsRender = false;
      }
    });
  }
  disposeObject(object) {
    this.needsRender = true;
    object.traverse((child) => {
      child.geometry?.dispose();
      if (child.material) {
        child.material.map?.dispose();
        child.material.dispose();
      }
    });
    object.removeFromParent();
  }
  reset(shape, mappings, canvases) {
    this.needsRender = true;
    for (const object of [...this.root.children]) this.disposeObject(object);
    this.shape = shape;
    this.meshes.clear();
    this.planes = [];
    this.root.position.set(...shape.map((n) => -(n - 1) / 2));
    this.markers = new THREE.Group();
    this.root.add(this.markers);
    const box = new THREE.LineSegments(
      new THREE.EdgesGeometry(new THREE.BoxGeometry(...shape)),
      new THREE.LineBasicMaterial({ color: 0x456077 }),
    );
    box.position.set(...shape.map((n) => (n - 1) / 2));
    this.root.add(box);
    const axes = new THREE.AxesHelper(Math.max(...shape) * 0.25);
    axes.position.set(-10, -10, -10);
    this.root.add(axes);
    for (let axis = 0; axis < 3; axis++) {
      const canvas = document.createElement("canvas");
      canvas.width = 64;
      canvas.height = 64;
      const c = canvas.getContext("2d");
      c.font = "bold 40px sans-serif";
      c.fillStyle = ["#ff8e82", "#a3eaa8", "#8aafff"][axis];
      c.fillText(String(axis), 20, 45);
      const label = new THREE.Sprite(
        new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(canvas), depthTest: false }),
      );
      label.position.copy(axes.position);
      label.position.setComponent(
        axis,
        label.position.getComponent(axis) + Math.max(...shape) * 0.29,
      );
      label.scale.set(18, 18, 1);
      this.root.add(label);
      const { h, v, fixed } = mappings[axis];
      const corners = [
        [0, 0],
        [1, 0],
        [1, 1],
        [0, 1],
      ].map(([u, w]) => {
        const p = [0, 0, 0];
        p[h] = u * shape[h] - 0.5;
        p[v] = w * shape[v] - 0.5;
        return p;
      });
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(corners.flat(), 3));
      geometry.setAttribute("uv", new THREE.Float32BufferAttribute([0, 1, 1, 1, 1, 0, 0, 0], 2));
      geometry.setIndex([0, 1, 2, 0, 2, 3]);
      geometry.computeVertexNormals();
      const texture = new THREE.CanvasTexture(canvases[axis]);
      texture.colorSpace = THREE.SRGBColorSpace;
      texture.magFilter = THREE.NearestFilter;
      const material = new THREE.MeshBasicMaterial({
        map: texture,
        side: THREE.DoubleSide,
        transparent: true,
        opacity: 0.34,
        depthWrite: false,
      });
      const plane = new THREE.Mesh(geometry, material);
      this.root.add(plane);
      this.planes.push({ plane, fixed, texture });
    }
    this.resetCamera();
  }
  resetCamera() {
    const n = Math.max(...this.shape);
    const fit = Math.max(1, 1 / this.camera.aspect);
    this.camera.position.set(n * 1.3 * fit, -n * 1.2 * fit, n * 1.4 * fit);
    this.controls.target.set(0, 0, 0);
    this.controls.update();
  }
  resize() {
    const w = this.container.clientWidth,
      h = this.container.clientHeight;
    if (!w || !h) return;
    const previousAspect = this.camera.aspect;
    this.camera.aspect = w / h;
    const ratio = Math.min(previousAspect, 1) / Math.min(this.camera.aspect, 1);
    this.camera.position.sub(this.controls.target).multiplyScalar(ratio).add(this.controls.target);
    this.needsRender = true;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(w, h);
  }
  updatePlanes(slices, visible) {
    this.needsRender = true;
    this.planes.forEach(({ plane, fixed, texture }) => {
      plane.position.setComponent(fixed, slices[fixed]);
      plane.visible = visible;
      texture.needsUpdate = true;
    });
  }
  removeSheet(id) {
    const mesh = this.meshes.get(id);
    if (mesh) this.disposeObject(mesh);
    this.meshes.delete(id);
  }
  setSheet(id, buffer, color, kind = "prompted") {
    this.needsRender = true;
    this.removeSheet(id);
    const header = new DataView(buffer),
      nv = header.getUint32(0, true),
      nf = header.getUint32(4, true);
    if (!nv) return;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(new Float32Array(buffer, 8, nv * 3), 3),
    );
    geometry.setIndex(new THREE.BufferAttribute(new Uint32Array(buffer, 8 + nv * 12, nf * 3), 1));
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      color,
      side: THREE.DoubleSide,
      roughness: 0.65,
      metalness: 0.08,
    });
    if (kind === "reference") {
      material.transparent = true;
      material.opacity = 0.32;
      material.depthWrite = false;
      material.polygonOffset = true;
      material.polygonOffsetFactor = -1;
      material.polygonOffsetUnits = -1;
    }
    const mesh = new THREE.Mesh(geometry, material);
    mesh.userData.kind = kind;
    this.root.add(mesh);
    this.meshes.set(id, mesh);
  }
  updatePoints(sheets) {
    this.needsRender = true;
    for (const child of [...this.markers.children]) this.disposeObject(child);
    for (const sheet of sheets.values())
      for (const point of sheet.points) {
        const marker = new THREE.Mesh(
          new THREE.SphereGeometry(2, 10, 8),
          new THREE.MeshBasicMaterial({ color: sheet.color, depthTest: false }),
        );
        marker.position.set(...point);
        marker.renderOrder = 10;
        this.markers.add(marker);
      }
  }
  showOverlay(key, visible) {
    this.needsRender = true;
    const mesh = this.meshes.get(key);
    if (mesh) mesh.visible = visible;
  }
  showSurfaces(visible) {
    this.needsRender = true;
    for (const mesh of this.meshes.values())
      if (mesh.userData.kind === "prompted") mesh.visible = visible;
  }
}
