"""Clawvatar CLI — start server, validate avatars, manage the engine."""

from __future__ import annotations

import argparse
import logging
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="clawvatar",
        description="Clawvatar Engine — real-time 3D avatar animation server",
    )
    sub = parser.add_subparsers(dest="command")

    # serve
    serve_p = sub.add_parser("serve", help="Start the WebSocket server")
    serve_p.add_argument("-c", "--config", default="clawvatar.yaml", help="Config file path")
    serve_p.add_argument("--host", default=None, help="Override server host")
    serve_p.add_argument("--port", type=int, default=None, help="Override server port")
    serve_p.add_argument("--avatar", default=None, help="Avatar model (.glb/.vrm) to auto-load")
    serve_p.add_argument("--ssl-cert", default=None, help="SSL certificate file for HTTPS")
    serve_p.add_argument("--ssl-key", default=None, help="SSL key file for HTTPS")

    # init
    init_p = sub.add_parser("init", help="Create a default config file")
    init_p.add_argument("-o", "--output", default="clawvatar.yaml", help="Output path")

    # validate
    val_p = sub.add_parser("validate", help="Validate a 3D avatar model file")
    val_p.add_argument("model", help="Path to .glb/.vrm/.gltf file")

    # info
    info_p = sub.add_parser("info", help="Show avatar model info")
    info_p.add_argument("model", help="Path to .glb/.vrm/.gltf file")

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.command == "serve":
        _cmd_serve(args)
    elif args.command == "init":
        _cmd_init(args)
    elif args.command == "validate":
        _cmd_validate(args)
    elif args.command == "info":
        _cmd_info(args)
    else:
        parser.print_help()
        sys.exit(1)


def _cmd_serve(args):
    import uvicorn

    from clawvatar.config import ClawvatarConfig
    from clawvatar.server import create_app

    config = ClawvatarConfig.from_yaml(args.config)
    if args.host:
        config.server.host = args.host
    if args.port:
        config.server.port = args.port
    if args.avatar:
        config.avatar.model_path = args.avatar

    create_app(config)
    uvicorn_kwargs = {
        "host": config.server.host,
        "port": config.server.port,
        "log_level": "info",
    }
    if args.ssl_cert and args.ssl_key:
        uvicorn_kwargs["ssl_certfile"] = args.ssl_cert
        uvicorn_kwargs["ssl_keyfile"] = args.ssl_key
    uvicorn.run("clawvatar.server:app", **uvicorn_kwargs)


def _cmd_init(args):
    from clawvatar.config import ClawvatarConfig

    config = ClawvatarConfig()
    config.to_yaml(args.output)
    print(f"Config written to {args.output}")
    print()
    print("Next steps:")
    print("  1. Get a 3D avatar (.glb or .vrm):")
    print("     - Avaturn: https://avaturn.me (free, from photo)")
    print("     - VRoid Studio: https://vroid.com/en/studio (free, customizable)")
    print("  2. Set avatar.model_path in config")
    print("  3. Run: clawvatar serve")


def _cmd_validate(args):
    from clawvatar.avatar.loader import AvatarLoader

    try:
        loader = AvatarLoader()
        avatar = loader.load(args.model)
        has_bs = "YES" if avatar.has_blend_shapes else "NO (lip-sync won't work!)"
        print(f"VALID — '{avatar.name}'")
        print(f"  Meshes: {len(avatar.meshes)}")
        print(f"  Vertices: {avatar.vertex_count}")
        print(f"  Faces: {avatar.face_count}")
        print(f"  Blend shapes: {has_bs}")
        if avatar.blend_shape_names:
            print(f"  Shape names: {', '.join(avatar.blend_shape_names[:10])}")
            if len(avatar.blend_shape_names) > 10:
                print(f"  ... and {len(avatar.blend_shape_names) - 10} more")
    except Exception as e:
        print(f"INVALID: {e}")
        sys.exit(1)


def _cmd_info(args):
    _cmd_validate(args)


if __name__ == "__main__":
    main()
