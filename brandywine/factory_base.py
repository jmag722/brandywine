from typing import Type
import brandywine.spatial_schemes.lax_friedrich as lf
import brandywine.spatial_schemes.lax_wendroff as lw

class Factory:
    @classmethod
    def create(self, children:list[Type], child_id:str, **kwargs):
        for child in children:
            if child_id.lower() in child.names():
                return child(**kwargs)
        raise KeyError(f"Child `{child_id}` is an invalid subclass identifier.")

class InviscidFluxFactory(Factory):
    child_classes = [
        lf.LaxFriedrichFlux,
        lw.LaxWendroffFlux,
    ]

    @classmethod
    def create(self, child_id:str, **kwargs):
        return super().create(self.child_classes, child_id, **kwargs)