from functools import partial
import traceback
from typing import Any, Callable, Coroutine, Dict, List, Optional, Union
import tensorpc

from tensorpc.apps.flow import flowapp

from tensorpc.apps.flow.flowapp import mui, three
import numpy as np
import inspect
from cumm import dtypes
from cumm.gemm.arch.tensorop import ALL_TENSOR_OP_MAP
from cumm.gemm.core.metaarray import MetaArray, seq


class WarpPanel(three.Group):

    def __init__(self, size: float,
                 on_click: Callable[[bool, int], three.CORO_NONE], 
                 on_reset: Callable[[], Coroutine[None, None, None]]) -> None:
        super().__init__({})
        layout: Dict[str, three.ToggleButton] = {}
        self.warp_btns: List[three.ToggleButton] = []
        self.on_reset = on_reset
        warp_colors = [ 
            "red", 
            "blue",
            "green", 
            "burlywood",
            "cyan", 
            "maroon",
            "chartreuse",
            "blueviolet",
            "gray", 
            "orangered",
            "steelblue",
            "palevioletred",
            "palegreen",
            "dodgerblue",
            "deeppink",
            "olive",
            "chocolate",
            "springgreen",
            "seagreen",
        ]
        for i in range(4):
            for j in range(8):
                idx = i * 8 + j
                color = warp_colors[idx % len(warp_colors)]
                btn = three.ToggleButton(str(idx), size, size,
                                         partial(self.on_click, index=idx))
                btn.props.position = (j * size * 1.05, -i * size * 1.05, 0)
                layout[f"{i}_{j}"] = btn
                btn.mesh.props.click_color = color
                self.warp_btns.append(btn)
        btn = three.Button("R", size, size, self.reset)
        btn.props.position = (0 * size * 1.05, -4 * size * 1.05, 0)
        layout[f"reset"] = btn

        self.add_layout(layout)
        self.layout = layout
        self.callback = on_click

    async def on_click(self, enable: bool, index: int):
        coro = self.callback(enable, index)
        if inspect.iscoroutine(coro):
            await coro

    async def reset(self, ev):
        for v in self.warp_btns:
            await v.mesh.set_checked(False)
        await self.on_reset()
        
    def get_callback(self) -> Optional[Callable]:
        return self.callback

    def set_callback(self, val: Any):
        self.callback = val
        for v in self.warp_btns:
            v.set_callback(partial(self.on_click, index=int(v.name)))
        return

class GemmParams:
    def __init__(self) -> None:
        self._dtypes = mui.Input("Dtype Shortcut", init="f16,f16,f16,f32,f32")
        self._trans_a = mui.Switch("TransA")
        self._trans_b = mui.Switch("TransB")
        self._trans_c = mui.Switch("TransC")

        self._tile_shape = mui.Input("Tile Shape")
        self._warp_tile_shape = mui.Input("Warp Tile Shape")
        self._tile_shape.prop(padding_bottom="5px")
        self._warp_tile_shape.prop(padding_bottom="5px")
        name_to_top = {v.short_repr(): v for v in ALL_TENSOR_OP_MAP.values()}
        self.name_to_top = name_to_top
        self._tensorop_select = mui.Select("TensorOp",
                                          [(k, k) for k in name_to_top.keys()])

    def get_layout(self):
        return {
            "ts": self._tile_shape,
            "wts": self._warp_tile_shape,
            "ta": self._trans_a,
            "tb": self._trans_b,
            "tc": self._trans_c,
            "top": self._tensorop_select,
        }

    def valid(self):
        try:
            dtype_abcac = [
                dtypes.get_dtype_by_shortcut(s.strip())
                for s in self._dtypes.value.split(",")
            ]
            assert len(dtype_abcac) == 5
            ts = seq(*self._tile_shape.json())
            wts = seq(*self._warp_tile_shape.json())
            assert len(ts) == 3 and len(wts) == 3
            wc = ts // wts
            return True 
        except:
            traceback.print_exc()
            return False 

    @property
    def tile_shape(self):
        return seq(*self._tile_shape.json())

    @property
    def warp_tile_shape(self):
        return seq(*self._warp_tile_shape.json())

    @property
    def trans_a(self):
        return self._trans_a.checked
    @property
    def trans_b(self):
        return self._trans_b.checked

    @property
    def trans_c(self):
        return self._trans_c.checked

    @property 
    def tensor_op(self):
        top = None
        if self._tensorop_select.value in self.name_to_top:
            top = self.name_to_top[self._tensorop_select.value]
        return top

class TensorOpViewer(three.Group):

    def __init__(self, params: GemmParams) -> None:
        super().__init__({})
        self.params = params
        self.operandA = three.Boxes2D(20000)
        
        self.operandA.prop(color="royalblue",
                           line_color="black",
                           alpha=0.0,
                           line_width=1,
                           hover_line_color="blue",
                           hover_line_width=2)

        self.operandB = three.Boxes2D(20000)
        self.operandB.prop(color="royalblue",
                           line_color="black",
                           alpha=0.0,
                           line_width=1,
                           hover_line_color="blue",
                           hover_line_width=2)

        self.operandC = three.Boxes2D(20000)
        self.operandC.prop(color="royalblue",
                           line_color="black",
                           alpha=0.0,
                           line_width=1,
                           hover_line_color="blue",
                           hover_line_width=2)
        self.warp_panel = WarpPanel(1, self.on_warp_select, self.on_reset)
        self.warp_panel.props.position = (0, -10, 0)
        self.html = three.Html({
                "box":
                mui.VBox({
                    "btn0": mui.Button("RTX", lambda: print("RTX")),
                }).prop(width=300)
            })
        self.html.prop(position=(-4, -6, 0), transform=True)
        self.add_layout({
            "matrix":
            three.Flex({
                "Inputs":
                three.HBox({
                    "AText":
                    three.FlexItem(three.Text("A").prop(color="blue", font_size=1)).prop(margin=0.1),
                    "A":
                    three.ItemBox({
                        "operandA": self.operandA,
                    }).prop(margin=0.1),
                    "BText":
                    three.ItemBox({
                        "B":
                        three.Text("B").prop(color="blue", font_size=1),
                    }).prop(margin=0.1),
                    "B":
                    three.ItemBox({
                        "operandB": self.operandB,
                    }).prop(margin=0.1),
                }).prop(padding=0.2),
                "Outputs":
                three.HBox({
                    "C":
                    three.ItemBox({
                        "operandC": self.operandC,
                    }).prop(margin=0.1),
                }).prop(padding=0.2),
                "reflow":
                three.FlexAutoReflow(),
            }).prop(flex_direction="column", position=(2, 6, 0)),
            "html0":self.html,
            "wraps":
            self.warp_panel.prop(position=(-8, 0, 0)),
        })

    async def on_warp_select(self, enable, index):
        if self.params.tensor_op is not None:
            top = self.params.tensor_op
            map_a = top.a_map()  # [Nfrag, 32, 2]
            map_b = top.b_map()
            map_c = top.c_map()
            a_coord = map_a[:, index]  # [Nfrag, 2]
            b_coord = map_b[:, index]  # [Nfrag, 2]
            c_coord = map_c[:, index]  # [Nfrag, 2]
            if enable:
                click_color = self.warp_panel.warp_btns[index].mesh.props.click_color
                if not isinstance(click_color, mui.Undefined):
                    color = (*mui.colors.str_to_rgb_float(click_color), 0.8)
                else:
                    color = (0.5, 0, 0, 0.8)
            else:
                color = (0, 0, 0, 0.0)
            self.operandA.props.colors.reshape(*top.shape_a, 4)[a_coord[:, 0],
                                                          a_coord[:,
                                                                  1]] = color
            self.operandB.props.colors.reshape(*top.shape_b, 4)[b_coord[:, 0],
                                                          b_coord[:,
                                                                  1]] = color
            self.operandC.props.colors.reshape(*top.shape_c, 4)[c_coord[:, 0],
                                                          c_coord[:,
                                                                  1]] = color
            await self.operandA.update_boxes(colors=self.operandA.props.colors)
            await self.operandB.update_boxes(colors=self.operandB.props.colors)
            await self.operandC.update_boxes(colors=self.operandC.props.colors)

    async def on_reset(self):
        if not isinstance(self.operandA.props.colors, mui.Undefined):
            self.operandA.props.colors[:, :3] = 0
            self.operandA.props.colors[:, 3] = 0.0
            await self.operandA.update_boxes(colors=self.operandA.props.colors)
        if not isinstance(self.operandB.props.colors, mui.Undefined):
            self.operandB.props.colors[:, :3] = 0
            self.operandB.props.colors[:, 3] = 0.0
            await self.operandB.update_boxes(colors=self.operandB.props.colors)
        if not isinstance(self.operandC.props.colors, mui.Undefined):
            self.operandC.props.colors[:, :3] = 0
            self.operandC.props.colors[:, 3] = 0.0
            await self.operandC.update_boxes(colors=self.operandC.props.colors)

    async def on_tensorop_select(self, value: three.ValueType):
        if value != "":
            top = self.params.name_to_top[str(value)]
            scale = 0.6
            a_shape = [top.shape[0], top.shape[2]]
            b_shape = [top.shape[2], top.shape[1]]
            c_shape = [top.shape[0], top.shape[1]]
            #
            centersAScalar = np.arange(top.km).astype(np.float32)
            centersA = np.stack(
                [-(centersAScalar // a_shape[1]), centersAScalar % a_shape[1]],
                axis=1)
            centersBScalar = np.arange(top.kn).astype(np.float32)
            centersB = np.stack(
                [-(centersBScalar // b_shape[1]), centersBScalar % b_shape[1]],
                axis=1)
            centersCScalar = np.arange(top.mn).astype(np.float32)
            centersC = np.stack(
                [-(centersCScalar // c_shape[1]), centersCScalar % c_shape[1]],
                axis=1)
            dimensions = np.ones((1, ), np.float32) * scale

            colorsA = np.zeros((top.km, 4), np.float32)
            colorsB = np.zeros((top.kn, 4), np.float32)
            colorsC = np.zeros((top.mn, 4), np.float32)
            colorsA[:, -1] = 0.0
            colorsB[:, -1] = 0.0
            colorsC[:, -1] = 0.0

            await self.operandA.update_boxes(centersA[:, ::-1] * scale,
                                             dimensions,
                                             colors=colorsA)
            await self.operandB.update_boxes(centersB[:, ::-1] * scale,
                                             dimensions,
                                             colors=colorsB)
            await self.operandC.update_boxes(centersC[:, ::-1] * scale,
                                             dimensions,
                                             colors=colorsC)
            print(value)

    def is_visible_undefined(self):
        return isinstance(self.props.visible, mui.Undefined)

    async def set_visible(self, visible: bool):
        await self.update_object3d(visible=visible)
        await self.html.update_object3d(visible=visible)



class GemmDesigner(flowapp.EditableLayoutApp):

    def __init__(self) -> None:
        super().__init__(False)
        self.root.props.min_height = 0
        self.root.props.min_width = 0
        self.set_init_window_size([800, 600])

    def app_create_layout(self) -> Dict[str, mui.MUIComponentType]:
        cam = three.OrthographicCamera(True, near=0.1, far=1000, zoom=50.0)
        cam.prop(position=(0, 0, 10), up=(0, 0, 1))
        ctrl = three.MapControl()
        ctrl.props.enable_rotate = False
        self.gemm_params = GemmParams()

        self.top_viewer = TensorOpViewer(self.gemm_params)
        self.canvas = three.ThreeCanvas({
            "cam": cam,
            "ctrl": ctrl,
            "top_viewer": self.top_viewer,
            # "tr": three.TransformControls().prop(object3d_uid="root.d3v.d3.top_viewer")
        })
        return {
            "d3v":
            mui.VBox({
                "d3":
                self.canvas,
                "hud":
                mui.VBox({
                    **self.gemm_params.get_layout(),
                    # "update": mui.Button("Box2d", self.on_box2d_update),
                    "sync": mui.Button("Set Gemm Params", self._set_gemm_params),

                    "btn3": mui.Button("Test", self._select_designer),
                }).prop(position="absolute",
                        top=0,
                        right=0,
                        z_index=5,
                        justify_content="flex-end")
            }).prop(position="relative", flex=1, min_height=0),
        }
        
    async def _set_gemm_params(self):
        if self.gemm_params.valid():
            await self.top_viewer.on_tensorop_select(self.gemm_params._tensorop_select.value)

    async def _select_designer(self):
        if self.top_viewer.is_visible_undefined():
            await self.top_viewer.set_visible(False)
        else:
            await self.top_viewer.set_visible(not self.top_viewer.props.visible)
    
    async def handle_code_editor_event(self, event: flowapp.AppEditorFrontendEvent):
        if event.type == flowapp.AppEditorFrontendEventType.Save:
            code = event.data
            print(code)
